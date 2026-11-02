#!/usr/bin/env python3
"""Truth-only moving collision geometry with a narrow handle contact allowance."""
import json
import time
import copy
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from moveit_msgs.msg import PlanningScene, AllowedCollisionEntry
from sorting_interfaces.srv import SetHandleContact
from publish_static_fixtures import scene_message
from publish_oracle_scene import update_scene
from m2b_common import config
from prediction_core import Observation, handle_position
from ownership_core import require

FINGERS = ("panda_leftfinger", "panda_rightfinger")


def contact_scene(template, contact, settings, allow):
    w,x,y,z = contact["tray_quat_wxyz"]; quat = (x,y,z,w)
    position = handle_position(contact["tray_xyz"], quat, settings["handle_offset_m"])
    observation = Observation(int(contact["reset_seq"]),1,round(contact["sim_time"]*1e9),tuple(position),(0,0,0),quat)
    scene = update_scene(template, observation, settings)
    acm = template.allowed_collision_matrix; old = acm.entry_names
    names = sorted(set(old)|set(FINGERS)|{"handle_bar"})
    permitted = {(a,b) for i,a in enumerate(old) for j,b in enumerate(old) if acm.entry_values[i].enabled[j]}
    if allow:
        permitted |= {(finger,"handle_bar") for finger in FINGERS}|{("handle_bar",finger) for finger in FINGERS}
    scene.allowed_collision_matrix.entry_names = names
    scene.allowed_collision_matrix.entry_values = [AllowedCollisionEntry(enabled=[a==b or (a,b) in permitted for b in names]) for a in names]
    return scene


class GraspScene(Node):
    def __init__(self):
        super().__init__("m2c_oracle_grasp_scene", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("enable_oracle", False)
        require(self.get_parameter("enable_oracle").value is True, "Grasp scene requires explicit oracle debug")
        self.settings = config(); self.template = scene_message(); self.contact = {}; self.owner = {}; self.allow = False
        self.received = 0.0
        self.publisher = self.create_publisher(PlanningScene, "/planning_scene", 1)
        self.state = self.create_publisher(String, "/sorting/debug/grasp_scene_state", 1)
        self.create_subscription(String, "/sorting/contact_state", self.receive_contact, 1)
        self.create_subscription(String, "/sorting/arm/owner_state", self.receive_owner, 1)
        self.create_service(SetHandleContact, "/sorting/debug/set_handle_contact", self.set_contact)
        self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def receive_contact(self, message):
        candidate = json.loads(message.data)
        if candidate.get("reset_seq",0) < self.contact.get("reset_seq",0): return
        if candidate.get("reset_seq",0) != self.contact.get("reset_seq",0): self.allow = False
        self.contact = candidate; self.received = time.monotonic()

    def receive_owner(self, message): self.owner = json.loads(message.data)

    def set_contact(self, request, response):
        try:
            require(time.monotonic()-self.received < 0.5, "stale_contact_state")
            require(request.episode_id == self.contact["reset_seq"], "episode_mismatch")
            require(request.owner_epoch == self.owner.get("epoch"), "owner_epoch_mismatch")
            require(self.owner.get("owner") in {"HOLD","SERVO"} and not self.owner.get("busy"), "owner_not_ready")
            if request.allow_handle_contact:
                bar = handle_position(self.contact["tray_xyz"], [self.contact["tray_quat_wxyz"][i] for i in (1,2,3,0)], self.settings["handle_offset_m"])
                hand = self.contact["hand_xyz"]
                require(self.contact["support_force_N"] > 0.5 and
                        abs(hand[0]-bar[0]) < 0.008 and abs(hand[1]-bar[1]) < 0.008 and
                        0.025 < hand[2]-bar[2]-self.settings["hand_above_handle_m"] < 0.09,
                        "enable_only_in_pregrasp_corridor")
            else:
                require(self.contact["left_force_N"]+self.contact["right_force_N"] < 0.1 and
                        min(self.contact["finger_positions_m"]) > 0.039, "release_fingers_before_disabling_allowance")
            self.allow = bool(request.allow_handle_contact); response.success = True; response.reason = "applied"
            self.tick()
        except Exception as error: response.success = False; response.reason = str(error)
        return response

    def tick(self):
        if not self.contact: return
        age = self.get_clock().now().nanoseconds/1e9-self.contact["sim_time"]
        valid = time.monotonic()-self.received < 0.5 and -0.02 <= age < 0.15
        if valid: self.publisher.publish(contact_scene(self.template,self.contact,self.settings,self.allow))
        self.state.publish(String(data=json.dumps({"valid": valid, "capture_time_s": self.contact["sim_time"],
                                                  "episode_id": self.contact["reset_seq"], "allow_handle_contact": self.allow,
                                                  "physics_attachment": False})))


def main():
    rclpy.init(); node = GraspScene()
    try: rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally: node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()
