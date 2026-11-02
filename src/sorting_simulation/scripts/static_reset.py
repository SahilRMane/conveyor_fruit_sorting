"""Restore and verify the home state before a nominal grasp trial."""
import json
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from controller_manager_msgs.srv import ListControllers, SwitchController
from mujoco_ros2_control_msgs.srv import ResetWorld, SetPause, StepSimulation
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

from test_static_lift import LiftTest, require, distance

ARM = "panda_arm_controller"
HAND = "panda_hand_controller"


def home_keyframe():
    path = Path(get_package_share_directory("sorting_description"))/"mujoco/static_grasp/panda_static.xml"
    key = ET.parse(path).getroot().find("keyframe/key[@name='home']")
    require(key is not None, "Installed Panda home keyframe is missing")
    expected = {"qpos": [float(v) for v in key.get("qpos", "").split()],
                "qvel": [float(v) for v in key.get("qvel", " ".join(["0"]*15)).split()],
                "ctrl": [float(v) for v in key.get("ctrl", "").split()]}
    require(len(expected["qpos"]) == 16 and len(expected["qvel"]) == 15 and len(expected["ctrl"]) == 8,
            "Home keyframe has unexpected dimensions")
    return expected


def validate_reset_record(record, expected):
    require(record.get("schema_version") == 1, "Update the contact monitor: reset record is missing or outdated")
    errors = {}
    for name in ("qpos", "qvel", "ctrl"):
        values = record.get(name, [])
        require(len(values) == len(expected[name]) and all(math.isfinite(v) for v in values),
                f"Invalid reset {name} values")
        errors[name] = max(abs(a-b) for a,b in zip(values, expected[name]))
        require(errors[name] < 1e-8, f"Reset did not restore {name}")
    for name in ("act_max_abs", "qfrc_applied_max_abs", "xfrc_applied_max_abs"):
        require(math.isfinite(record[name]) and abs(record[name]) < 1e-8,
                f"Reset did not clear {name}")
    return errors


class ResetObserver(LiftTest):
    def __init__(self, node, plan):
        self.reset_record = {}
        self.joint_velocities = {}
        self.joint_updates = 0
        super().__init__(node, plan)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, "/sorting/reset_state", self.receive_reset, qos)

    def receive_reset(self, message):
        record = json.loads(message.data)
        if record.get("reset_seq", -1) >= self.reset_record.get("reset_seq", -1):
            self.reset_record = record

    def receive_joints(self, message):
        self.joints = dict(zip(message.name, message.position))
        self.joint_velocities = dict(zip(message.name, message.velocity))
        self.joint_updates += 1

    def prepare_for_reset(self):
        self.guarded = False
        self.stage = "reset"
        self.reference = None
        self.grasp = None
        self.maximum_slip = 0.0
        self.maximum_placement_force = 0.0
        self.history.clear()
        self.joints.clear()
        self.joint_velocities.clear()


class ResetManager:
    def __init__(self, observer):
        self.test = observer
        self.node = observer.node
        self.expected = home_keyframe()
        self.clients = {}
        self.list_client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        self.switch_client = self.node.create_client(SwitchController, "/controller_manager/switch_controller")

    def client(self, kind):
        if kind not in self.clients:
            type_name = f"mujoco_ros2_control_msgs/srv/{kind.__name__}"
            found = []
            def discover():
                found[:] = [name for name, types in self.node.get_service_names_and_types() if type_name in types]
                return len(found) > 0
            self.test.until(discover, 10.0)
            require(len(found) == 1, f"Expected one {kind.__name__} service")
            self.clients[kind] = self.node.create_client(kind, found[0])
        return self.clients[kind]

    def call(self, client, request, timeout=10.0):
        self.test.until(client.service_is_ready, timeout)
        return self.test.wait_future(client.call_async(request), timeout)

    def pause(self, paused):
        request = SetPause.Request()
        request.paused = paused
        response = self.call(self.client(SetPause), request)
        require(response.success, response.message)

    def states(self):
        response = self.call(self.list_client, ListControllers.Request())
        return {controller.name: controller.state for controller in response.controller}

    def advance(self, steps):
        request = StepSimulation.Request()
        request.steps = steps
        response = self.call(self.client(StepSimulation), request, 30.0)
        require(response.success, response.message)

    def switch_while_paused(self, activate, deactivate):
        if not activate and not deactivate:
            return
        self.test.until(self.switch_client.service_is_ready)
        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = False
        request.timeout.sec = 5
        future = self.switch_client.call_async(request)
        deadline = time.monotonic() + 10.0
        # A controller switch needs a control update. Small physics batches supply
        # clock ticks while the simulation remains paused between batches.
        while not future.done():
            require(time.monotonic() < deadline, "Paused controller switch timed out")
            self.advance(10)
        response = future.result()
        require(response.ok, response.message)

    def reset_home(self):
        report = {"status": "FAIL", "keyframe": "home"}
        self.test.prepare_for_reset()
        try:
            # Find required services before changing controller state.
            for kind in (SetPause, ResetWorld, StepSimulation):
                self.client(kind)
            self.pause(True)
            before_states = self.states()
            require(before_states.get("joint_state_broadcaster") == "active", "Joint state broadcaster must be active")
            require(all(before_states.get(n) in {"active", "inactive"} for n in (ARM, HAND)),
                    "Arm and hand controllers must be configured")
            deactivate = [n for n in (ARM, HAND) if before_states[n] == "active"]
            self.switch_while_paused([], deactivate)
            inactive = self.states()
            require(all(inactive.get(n) == "inactive" for n in (ARM, HAND)),
                    "Both joint controllers must be inactive before reset")
            self.advance(10)
            self.test.until(lambda: bool(self.test.state), 10.0)
            require("reset_seq" in self.test.state, "Rebuild and restart the launch to load the M1C4 contact monitor")
            before_seq = max(self.test.state.get("reset_seq", 0), self.test.reset_record.get("reset_seq", 0))
            request = ResetWorld.Request()
            request.keyframe = "home"
            response = self.call(self.client(ResetWorld), request)
            require(response.success, response.message)
            self.advance(10)
            self.switch_while_paused([ARM, HAND], [])
            active = self.states()
            require(all(active.get(n) == "active" for n in (ARM, HAND)), "Controllers did not reactivate")
            joint_updates = self.test.joint_updates
            self.pause(False)
            self.test.until(lambda: self.test.state.get("reset_seq", 0) > before_seq
                            and self.test.reset_record.get("reset_seq") == self.test.state.get("reset_seq")
                            and self.test.joint_updates > joint_updates, 10.0)
            raw = self.test.reset_record.copy()
            errors = validate_reset_record(raw, self.expected)
            self.test.wait_sim(2.0)
            state = self.test.state.copy()
            require(state.get("friction_cone") == 1 and state.get("friction_impedance_ratio") == 100,
                    "Use the validated lift scene")
            require(abs(state["tray_mass_kg"]-self.test.plan["initial_state"]["tray_mass_kg"]) < 1e-6,
                    "Tray mass differs from the validated plan")
            require(distance(state["tray_xyz"], self.test.plan["initial_state"]["tray_xyz"]) < 0.001,
                    "Tray did not return to its nominal rest pose")
            dot = abs(sum(a*b for a,b in zip(state["tray_quat_wxyz"], self.test.plan["initial_state"]["tray_quat_wxyz"])))
            require(dot > math.cos(math.radians(1)/2), "Tray orientation did not reset")
            home = self.expected["qpos"][:7]
            require(all(n in self.test.joints for n in self.test.names), "Missing arm joint states")
            require(all(math.isfinite(self.test.joints[n]) for n in self.test.names), "Invalid arm joint positions")
            arm_error = max(abs(self.test.joints[n]-q) for n,q in zip(self.test.names, home))
            require(arm_error < 0.02, "Arm did not remain at home after reactivation")
            require(max(abs(q-0.04) for q in state["finger_positions_m"]) < 0.001,
                    "Fingers did not reset to the open position")
            required = self.test.names + ["panda_finger_joint1", "panda_finger_joint2"]
            require(all(n in self.test.joint_velocities and math.isfinite(self.test.joint_velocities[n]) for n in required),
                    "Missing or invalid joint velocities")
            require(max(abs(self.test.joint_velocities[n]) for n in required) < 0.01,
                    "Robot did not settle after reset")
            require(state["forbidden_contacts"] == 0 and state["left_force_N"] + state["right_force_N"] < 0.1,
                    "Unexpected contact after reset")
            require(abs(state["support_force_N"]-self.test.plan["initial_state"]["support_force_N"]) < 0.5,
                    "Tray did not settle on the support after reset")
            self.test.active_goal = None
            self.test.reference = None
            self.test.grasp = None
            self.test.stage = "startup"
            self.test.maximum_slip = 0.0
            self.test.maximum_placement_force = 0.0
            self.test.history.clear()
            self.test.guarded = True
            self.test.check(state)
            self.test.confirm_controller()
            report.update(status="PASS", reset_seq=raw["reset_seq"], raw_record=raw,
                          raw_errors=errors, settled_state=state, arm_error_rad=arm_error,
                          inactive_controllers=inactive, active_controllers=active)
        except Exception as error:
            report["error"] = str(error)
            report["failure_handling"] = self.test.stop_after_failure()
        return report
