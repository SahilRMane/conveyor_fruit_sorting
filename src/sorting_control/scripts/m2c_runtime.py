"""Moving physical grip/lift through the existing exclusive arm gate."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.time import Time as RosTime
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from sorting_interfaces.srv import SetHandleContact
from std_msgs.msg import String
from std_srvs.srv import SetBool
from m2b_runtime import M2BProbe
from m2b_common import xyz, xyzw, timestamp
from prediction_core import predict, angle_error, tracking_twist
from grasp_core import (ClosureWindow, VelocityWindow, exit_deadline, smooth, braking_x,
                        constraint_require, grasp_slip_metrics, require_grasp_slip)
from ownership_core import NAMES, ns, require
from m2c_reset import ResetManager


class MovingGrasp(M2BProbe):
    def __init__(self, node):
        self.c = yaml.safe_load((Path(get_package_share_directory("sorting_control"))/"config/m2c.yaml").read_text())
        self.phase = "startup"; self.target_required = True; self.grasp = None; self.maximum_slip = 0.0
        self.contact_wall = 0.0; self.grasp_scene = {}; self.reset_record = {}; self.joint_updates = 0
        self.held = False; self.trace = []; self.last_tracking = {}; self.velocity_window = VelocityWindow()
        self.grip_trace = []; self.maximum_world_offset_change = 0.0
        self.resetting_local = False; self.hand_handle = None; self.hand_result = None; self.history = []
        super().__init__(node)
        self.settings["maximum_linear_speed_m_s"] = self.c["maximum_servo_linear_speed_m_s"]
        self.settings["maximum_angular_speed_rad_s"] = self.c["maximum_servo_angular_speed_rad_s"]
        self.settings["maximum_observation_age_s"] = self.c["maximum_target_age_s"]
        self.settings["maximum_source_wall_age_s"] = self.c["maximum_target_wall_age_s"]
        self.settings["position_gain_per_s"] = self.c["position_gain_per_s"]
        self.settings["orientation_gain_per_s"] = self.c["orientation_gain_per_s"]
        self.hand = ActionClient(node, GripperCommand, "/panda_hand_controller/gripper_cmd")
        self.contact_service = node.create_client(SetHandleContact, "/sorting/debug/set_handle_contact")
        self.reset_guard_client = node.create_client(SetBool, "/sorting/arm/reset_guard")
        node.create_subscription(String, "/sorting/debug/grasp_scene_state", self.receive_scene, 1)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, "/sorting/reset_state", self.receive_reset, qos)
        self.names = NAMES; self.active_goal = None; self.maximum_placement_force = 0.0; self.stage = "startup"
        self.plan = {"initial_state": {}}
        self.reset_manager = ResetManager(self)

    @property
    def state(self): return self.contact
    @property
    def joints(self): return self.q
    @property
    def joint_velocities(self): return self.v

    def receive_reset(self, message):
        value = json.loads(message.data)
        if value.get("reset_seq",0) >= self.reset_record.get("reset_seq",0): self.reset_record = value

    def receive_scene(self, message): self.grasp_scene = json.loads(message.data)

    def receive_joints(self, message):
        stamp = ns(message.header.stamp)
        if stamp <= self.q_stamp_ns: return
        self.q = dict(zip(message.name,message.position)); self.v = dict(zip(message.name,message.velocity))
        self.q_stamp_ns = stamp; self.joint_updates += 1
        if self.guarded and all(n in self.q and n in self.v for n in NAMES):
            values = [self.q[n] for n in NAMES]
            constraint_require(all(math.isfinite(x) for x in values+[self.v[n] for n in NAMES]), "Invalid arm feedback")
            if self.previous_q is not None:
                step = max(abs(a-b) for a,b in zip(values,self.previous_q)); self.maximum_joint_step = max(self.maximum_joint_step,step)
                constraint_require(step < 0.02, "Arm feedback step exceeded 0.02 rad")
            self.previous_q = values
            speed = max(abs(self.v[n]) for n in NAMES); self.maximum_joint_speed = max(self.maximum_joint_speed,speed)
            constraint_require(speed < self.c["maximum_joint_speed_rad_s"], "M2C joint speed exceeded its 0.5 rad/s bound")

    def ensure_oracle_current(self):
        if self.target_required: super().ensure_oracle_current()
        if self.resetting_local: return
        require(time.monotonic()-self.contact_wall < 0.5, "Contact stream stopped")
        require(self.grasp_scene.get("valid") is True and
                -0.02 <= self.now_ns()/1e9-self.grasp_scene.get("capture_time_s",0) < 0.2,
                "Moving collision geometry is stale")

    def receive_contact(self, message):
        value = json.loads(message.data)
        if value.get("reset_seq",0) < self.contact.get("reset_seq",0): return
        if value.get("reset_seq",0) == self.contact.get("reset_seq",0) and value["sim_time"] < self.contact.get("sim_time",0): return
        self.contact = value; self.contact_wall = time.monotonic(); self.history.append(value)
        if self.guarded: self.check(value)

    def check(self, value):
        require("hand_linear_velocity_m_s" in value, "Rebuild sorting_simulation and restart to load the M2C velocity monitor")
        constraint_require(value["forbidden_contacts"] == 0, "Forbidden physical contact")
        require(all(math.isfinite(x) for x in value["tray_xyz"]+value["hand_xyz"]+value["hand_linear_velocity_m_s"]+
                    value["tray_linear_velocity_m_s"]+value["finger_positions_m"]), "Non-finite physical state")
        constraint_require(max(value["left_force_N"],value["right_force_N"]) < self.c["maximum_contact_force_N"], "Finger force exceeds 15 N")
        require(abs(value["gripper_force_limit_N"]-20.0) < 1e-6, "Use the physical 20 N gripper limit")
        constraint_require(abs(value["external_force_x_N"]) <= 8.0001, "Transport force exceeds 8 N")
        if self.phase in {"startup","planning","pregrasp","track"}:
            require(min(value["finger_positions_m"]) > 0.039 and value["left_force_N"]+value["right_force_N"] < 0.1,
                    "Fingers must stay open before closure authorization")
        if self.grasp is not None:
            metrics = grasp_slip_metrics(value,self.grasp)
            self.maximum_slip = max(self.maximum_slip,metrics["relative_slip_m"])
            self.maximum_world_offset_change = max(self.maximum_world_offset_change,metrics["world_offset_change_m"])
            self.grip_trace.append(dict(metrics,sim_time=value["sim_time"],phase=self.phase))
            require_grasp_slip(metrics,self.c["maximum_slip_m"])
            require(min(value["left_force_N"],value["right_force_N"]) > self.c["minimum_contact_force_N"], "Physical finger contact was lost")
            a,b = value["tray_quat_wxyz"],self.grasp["tray_quat_wxyz"]
            require(abs(sum(x*y for x,y in zip(a,b))) > math.cos(self.c["maximum_tilt_rad"]/2), "Tray tilt exceeds 3 degrees")
            rise = value["tray_xyz"][2]-self.grasp["tray_xyz"][2]
            require(-0.003 <= rise < self.c["maximum_lift_rise_m"], "Tray left the pickup height bounds")
            if rise > 0.005:
                require(value["support_force_N"] < self.c["maximum_support_force_lifted_N"], "Lifted tray retains support force")
                require(abs(value["external_force_x_N"]) < 1e-6, "Transport force did not disengage after lift")

    def hand_sample(self):
        transform = self.buffer.lookup_transform("world","panda_hand",RosTime())
        stamp = ns(transform.header.stamp); age = (self.now_ns()-stamp)/1e9
        require(-0.02 <= age < self.c["maximum_tf_age_s"], "Invalid hand transform age")
        p = np.asarray(xyz(transform.transform.translation)); quat = xyzw(transform.transform.rotation)
        velocity = self.velocity_window.observe(stamp,p)
        return p,quat,stamp,age,velocity

    def send(self, epoch=None):
        if not self.input_enabled: return
        self.ensure_oracle_current()
        p,quat,stamp,age,velocity = self.hand_sample()
        corrected = p+velocity*max(0.0,age) if velocity is not None else p
        target,feedforward,target_quat = self.command_function(self.now_ns())
        linear,angular = tracking_twist(target,feedforward,target_quat,corrected,quat,self.settings)
        from sorting_interfaces.msg import ArmServoCommand
        message = ArmServoCommand(); message.epoch = int(self.input_epoch); message.deadline = timestamp(self.input_deadline)
        message.command_type = ArmServoCommand.TWIST; message.twist.header.frame_id = "world"; message.twist.header.stamp = self.node.get_clock().now().to_msg()
        message.twist.twist.linear.x,message.twist.twist.linear.y,message.twist.twist.linear.z = map(float,linear)
        message.twist.twist.angular.x,message.twist.twist.angular.y,message.twist.twist.angular.z = map(float,angular)
        self.publisher.publish(message)
        self.last_tracking = {"time_ns":self.now_ns(),"tf_stamp_ns":stamp,"tf_age_s":age,
                              "hand_xyz":corrected.tolist(),"target_xyz":list(map(float,target)),
                              "position_error_m":(np.asarray(target)-corrected).tolist(),
                              "orientation_error_rad":angle_error(target_quat,quat),
                              "hand_velocity_m_s":velocity.tolist() if velocity is not None else None,
                              "target_velocity_m_s":list(map(float,feedforward)),"phase":self.phase}
        self.trace.append(copy.deepcopy(self.last_tracking))

    def allow_contact(self, allow):
        self.until(lambda: self.contact and self.owner and self.grasp_scene.get("valid"))
        request = SetHandleContact.Request(episode_id=int(self.contact["reset_seq"]), owner_epoch=int(self.owner["epoch"]),
                                           allow_handle_contact=bool(allow))
        for _ in range(5):
            response = self.call(self.contact_service,request)
            if response.success: break
            require(response.reason in {"owner_epoch_mismatch","owner_not_ready"}, response.reason); self.wait_sim(0.05)
        require(response.success,response.reason)
        for client in self.scene_clients:
            end = time.monotonic()+5.0
            while True:
                req = GetPlanningScene.Request(); req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                acm = self.call(client,req).scene.allowed_collision_matrix
                names = acm.entry_names
                verified = all(n in names for n in ("handle_bar","panda_leftfinger","panda_rightfinger"))
                if verified:
                    i = names.index("handle_bar")
                    verified = all(bool(acm.entry_values[i].enabled[names.index(f)])==allow for f in ("panda_leftfinger","panda_rightfinger"))
                if verified: break
                require(time.monotonic() < end,"Collision allowance was not committed in both scenes"); self.wait_sim(0.05)

    def close_hand(self):
        goal = GripperCommand.Goal(); goal.command.position = 0.0; goal.command.max_effort = 20.0
        self.until(self.hand.server_is_ready)
        self.hand_handle = self.future(self.hand.send_goal_async(goal)); require(self.hand_handle.accepted,"Hand rejected closure")
        self.hand_result = self.hand_handle.get_result_async()

    def run_pickup(self, folder):
        result = {"status":"FAIL","oracle_only":True,"physical_grasp":True,"forced_attachment":False,"checks":{},
                  "slip_frame":"panda_hand","slip_metric_version":2}
        self.attempt_record = result
        self.phase = "planning"; self.grasp = None; self.held = False; self.target_required = True
        self.trace.clear(); self.history.clear(); self.maximum_slip = 0; self.velocity_window = VelocityWindow()
        self.grip_trace.clear(); self.maximum_world_offset_change = 0.0
        self.reference = self.contact.copy(); self.plan["initial_state"] = self.reference.copy()
        self.phase = "pregrasp"
        pre = np.asarray(self.oracle.position)+np.array([0,0,self.settings["hand_above_handle_m"]+self.settings["pregrasp_clearance_m"]])
        result["pregrasp_plan"] = self.execute_plan(pre,self.settings["grasp_quaternion_xyzw"],folder,"pregrasp")
        result["pregrasp_refinement"] = self.refine_pose(pre,self.settings["grasp_quaternion_xyzw"])
        result["checks"]["planned_pregrasp"] = "PASS"
        self.acquire_mode("SERVO",seconds=self.c["motion_lease_s"],kind=1)
        self.lease_deadline = self.input_deadline
        self.command_function = lambda now: (pre,[0.0]*3,self.settings["grasp_quaternion_xyzw"])
        self.input_enabled = True; self.allow_contact(True)
        self.belt_speed = self.c["belt_speed_m_s"]; self.allow_belt = True; self.send_belt()
        self.phase = "track"; begin = self.now_ns(); window = ClosureWindow(self.c)
        def moving_reference(now):
            estimate = predict(self.oracle,now,now,self.settings)
            t = (now-begin)/1e9; blend,dblend = smooth(t,self.c["descend_duration_s"])
            target = np.asarray(estimate["position"]); target[2] += self.settings["hand_above_handle_m"]+0.06*(1-blend)
            velocity = np.asarray(self.oracle.velocity).copy(); velocity[2] -= 0.06*dblend
            return target,velocity,self.settings["grasp_quaternion_xyzw"]
        self.command_function = moving_reference
        def closure_ready():
            sample = self.last_tracking
            if not sample or sample["phase"] != "track" or sample["hand_velocity_m_s"] is None: return False
            deadline = exit_deadline(self.oracle,self.now_ns(),self.c,self.settings["safe_x_interval_m"][1])
            self.input_deadline = min(self.lease_deadline,deadline)
            ready = dict(sample,relative_velocity_m_s=(np.asarray(sample["hand_velocity_m_s"])-np.asarray(self.oracle.velocity)).tolist(),
                         observation_age_s=(self.now_ns()-self.oracle.stamp_ns)/1e9,
                         remaining_s=(deadline-self.now_ns())/1e9,oracle_only=True,
                         at_grasp_height=(self.now_ns()-begin)/1e9 >= self.c["descend_duration_s"])
            result["last_closure_check"] = ready
            return window.observe(ready)
        self.until(closure_ready,timeout=self.c["tracking_timeout_s"])
        result["closure"] = copy.deepcopy(result["last_closure_check"])
        result["closure"]["true_relative_velocity_m_s"] = (np.asarray(self.contact["hand_linear_velocity_m_s"])-np.asarray(self.contact["tray_linear_velocity_m_s"])).tolist()
        result["closure"]["true_position_error_m"] = (np.asarray(self.contact["hand_xyz"])-
              (np.asarray(self.contact["tray_xyz"])+np.array([0,0,0.117+0.1029]))).tolist()
        result["checks"]["moving_closure_authorized"] = "PASS"
        self.phase = "close"; self.close_hand(); stable = [None]
        def bilateral():
            good = min(self.contact["left_force_N"],self.contact["right_force_N"]) > self.c["minimum_contact_force_N"]
            if not good: stable[0] = None; return False
            if stable[0] is None: stable[0] = self.contact["sim_time"]
            return self.contact["sim_time"]-stable[0] >= self.c["contact_stable_span_s"]
        self.until(bilateral,timeout=self.c["contact_timeout_s"])
        self.grasp = copy.deepcopy(self.contact); self.target_required = False; self.held = True
        self.input_deadline = self.lease_deadline  # Held-tray phases use the motion lease, not belt exit.
        result["grasp_state"] = copy.deepcopy(self.grasp); result["checks"]["bilateral_physical_contact"] = "PASS"
        self.phase = "lift"; begin = self.now_ns(); start = np.asarray(self.grasp["hand_xyz"]); speed = self.c["belt_speed_m_s"]
        brake = {"time":None,"x":None}
        def lift_reference(now):
            t = (now-begin)/1e9; blend,dblend = smooth(t,self.c["lift_duration_s"])
            x = start[0]+speed*t; vx = speed
            if self.contact["tray_xyz"][2]-self.grasp["tray_xyz"][2] > 0.005 and brake["time"] is None:
                brake.update(time=t,x=x)
            if brake["time"] is not None: x,vx = braking_x(brake["x"],speed,t-brake["time"],self.c["horizontal_brake_duration_s"])
            return [x,start[1],start[2]+self.c["lift_height_m"]*blend],[vx,0,self.c["lift_height_m"]*dblend],self.settings["grasp_quaternion_xyzw"]
        self.command_function = lift_reference
        self.until(lambda: (self.now_ns()-begin)/1e9 >= self.c["lift_duration_s"] and
                   self.contact["tray_xyz"][2]-self.grasp["tray_xyz"][2] >= self.c["minimum_lift_rise_m"] and
                   self.contact["support_force_N"] < 0.1 and abs(self.contact["external_force_x_N"]) < 1e-6,timeout=15.0)
        require(self.hand_result.done(),"Gripper action did not finish during lift")
        hand_result = self.hand_result.result()
        require(hand_result.status == GoalStatus.STATUS_SUCCEEDED and hand_result.result.stalled and not hand_result.result.reached_goal,
                "Hand must report a stalled handle grip")
        result["gripper_result"] = {"stalled":hand_result.result.stalled,"reached_goal":hand_result.result.reached_goal,"position_m":hand_result.result.position}
        result["checks"]["physical_lift_and_transport_disengagement"] = "PASS"
        self.phase = "hold"; held_before = len(self.history); self.wait_sim(self.c["hold_duration_s"])
        held_samples = self.history[held_before:]
        require(held_samples and min(s["tray_xyz"][2]-self.grasp["tray_xyz"][2] for s in held_samples) >= self.c["minimum_lift_rise_m"],
                "Tray did not remain lifted during hold")
        result["held_state"] = copy.deepcopy(self.contact); result["checks"]["physical_hold"] = "PASS"
        self.phase = "final_hold"; self.input_enabled = False; self.acquire_mode("HOLD")
        require(self.contact["tray_xyz"][2]-self.grasp["tray_xyz"][2] > 0.03 and self.contact["support_force_N"] < 0.1,
                "Tray was not retained after measured-state HOLD")
        require(self.owner["stats"]["overlap_violations"] == 0,"Arm source overlap")
        result["checks"]["exclusive_final_hold"] = "PASS"
        result.update(status="PASS",acceptance_passed=True,passed_checks=6,maximum_relative_slip_m=self.maximum_slip,
                      maximum_world_offset_change_m=self.maximum_world_offset_change,
                      final_grasp_metrics=grasp_slip_metrics(self.contact,self.grasp),
                      final_hold_tray_drop_m=result["held_state"]["tray_xyz"][2]-self.contact["tray_xyz"][2],
                      final_owner_state=copy.deepcopy(self.owner),final_contact_state=copy.deepcopy(self.contact))
        return result

    def wait_future(self, future, timeout=30.0): return self.future(future,timeout)
    def confirm_controller(self):
        require(self.owner.get("controller_streaming_verified") is True,"Controller streaming parameters are not verified")

    def prepare_for_reset(self):
        self.resetting_local = True; self.guarded = False; self.target_required = False; self.phase = "reset"
        self.reference = self.grasp = None; self.held = False; self.previous_q = None
        self.q.clear(); self.v.clear(); self.q_stamp_ns = 0; self.velocity_window = VelocityWindow()
        self.maximum_slip = 0; self.maximum_placement_force = 0; self.history.clear()

    def reset_episode(self):
        self.input_enabled = False; self.belt_speed = 0.0; self.send_belt(); self.target_required = False
        if self.owner.get("owner") != "HOLD": self.safe_hold()
        response = self.call(self.reset_guard_client,SetBool.Request(data=True))
        require(response.success,response.message)
        report = self.reset_manager.reset_home()
        require(report["status"] == "PASS", "Guarded reset failed: "+report.get("error","unknown"))
        self.resetting_local = False; self.phase = "startup"; self.guarded = True
        response = self.call(self.reset_guard_client,SetBool.Request(data=False))
        require(response.success,response.message)
        self.until(lambda: self.owner.get("owner")=="HOLD" and not self.owner.get("busy") and self.owner.get("ready")
                   and self.oracle and self.oracle.episode == self.contact["reset_seq"] and self.prediction and self.prediction.valid
                   and self.prediction.episode_id == self.contact["reset_seq"]
                   and self.grasp_scene.get("episode_id") == self.contact["reset_seq"])
        require(self.grasp_scene.get("allow_handle_contact") is False,"Contact allowance did not clear on reset")
        self.target_required = True; self.reference = self.contact.copy(); self.plan["initial_state"] = self.contact.copy()
        return report

    def stop_after_failure(self):
        from mujoco_ros2_control_msgs.srv import SetPause
        self.input_enabled = False; self.guarded = False; self.target_required = False; self.belt_speed = 0.0; self.send_belt()
        record = {"automatic_open_command":False,"paused":False}
        try: self.safe_hold(); record["hold"] = "PASS"
        except Exception as error: record["hold_error"] = str(error)
        try:
            response = self.reset_manager.call(self.reset_manager.client(SetPause),SetPause.Request(paused=True))
            require(response.success,response.message); record["paused"] = True
        except Exception as error: record["pause_error"] = str(error)
        return record
