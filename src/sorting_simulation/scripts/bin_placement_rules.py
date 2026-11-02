"""Physical placement checks. This module does not need a ROS node."""
import math


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def footprint_inside(state, plan):
    w, x, y, z = state["tray_quat_wxyz"]
    require(abs(w*w+x*x+y*y+z*z-1.0) < 1e-5, "Invalid tray quaternion")
    row_x = (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))
    row_y = (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))
    index = plan["bin_index"]
    centre = state["bin_floor_xyz"][index]
    half = state["bin_floor_half_size"][index]
    margin = plan["bin_wall_thickness_m"] + 0.003
    hx, hy, hz = plan["tray_half_size_m"]
    for cx in (-hx, hx):
        for cy in (-hy, hy):
            for cz in (-hz, hz):
                corner = (cx, cy, cz)
                px = state["tray_xyz"][0] + sum(a*b for a, b in zip(row_x, corner))
                py = state["tray_xyz"][1] + sum(a*b for a, b in zip(row_y, corner))
                if abs(px-centre[0]) > half[0]-margin or abs(py-centre[1]) > half[1]-margin:
                    return False
    return True


def check_state(state, plan, reference, grasp, stage):
    require(state.get("contact_schema_version") == 2 and state.get("bins_present") == [1, 1],
            "Rebuild the monitor and start panda_bins.launch.py")
    values = []
    for key in ("tray_xyz", "hand_xyz", "tray_quat_wxyz", "hand_quat_wxyz", "finger_positions_m",
                "bin_floor_force_N", "bin_wall_force_N", "bin_floor_top_z",
                "tray_linear_velocity_m_s", "tray_angular_velocity_rad_s"):
        values.extend(state[key])
    for key in ("sim_time", "left_force_N", "right_force_N", "support_force_N", "tray_floor_force_N",
                "external_force_x_N", "external_torque_y_Nm", "tray_mass_kg"):
        values.append(state[key])
    for key in ("bin_floor_xyz", "bin_floor_half_size"):
        for vector in state[key]:
            values.extend(vector)
    require(all(math.isfinite(value) for value in values), "Non-finite contact state")
    require(state["forbidden_contacts"] == 0, "Unexpected robot contact")
    require(state["tray_floor_force_N"] < 0.1, "Tray touched the ground plane")
    require(sum(state["bin_wall_force_N"]) < 0.1, "Tray touched a bin wall")
    require(max(state["left_force_N"], state["right_force_N"]) < 15.0, "Finger force exceeded 15 N")
    require(abs(state["gripper_force_limit_N"]-20.0) < 1e-6, "Unexpected gripper force limit")
    require(state.get("friction_cone") == 1 and state.get("friction_impedance_ratio") == 100,
            "Use the bin scene friction settings")
    require(abs(state["external_force_x_N"]) < 1e-9 and abs(state["external_torque_y_Nm"]) < 1e-9,
            "Stationary placement must not use conveyor force")
    require(abs(state["tray_mass_kg"]-plan["initial_state"]["tray_mass_kg"]) < 1e-6,
            "Tray mass differs from the native plan")
    for index in range(2):
        for key in ("bin_floor_xyz", "bin_floor_half_size"):
            require(distance(state[key][index], plan["initial_state"][key][index]) < 1e-6,
                    "Bin geometry differs from the native plan")
        require(abs(state["bin_floor_top_z"][index]-plan["initial_state"]["bin_floor_top_z"][index]) < 1e-6,
                "Bin floor height differs from the native plan")
    if reference is None:
        return 0.0
    index = plan["bin_index"]
    weight = reference["support_force_N"]
    dot = abs(sum(a*b for a, b in zip(state["tray_quat_wxyz"], reference["tray_quat_wxyz"])))
    require(dot > math.cos(math.radians(3)/2), "Tray tilt exceeded 3 degrees")
    target = plan["target_tray_xyz"]
    require(min(reference["tray_xyz"][2], target[2])-0.003 <= state["tray_xyz"][2]
            <= reference["tray_xyz"][2]+plan["lift_height_m"]+0.01, "Tray height outside transfer bounds")
    require(state["bin_floor_force_N"][1-index] < 0.1, "Tray touched the wrong bin floor")
    if stage in {"approach", "grip"}:
        require(distance(state["tray_xyz"], reference["tray_xyz"]) < 0.003
                and state["support_force_N"] > 0.5, "Tray left its initial support")
        require(state["bin_floor_force_N"][index] < 0.1, "Unexpected bin contact before transfer")
    slip = 0.0
    if grasp is not None and stage in {"lift", "transfer", "lower"}:
        require(min(state["left_force_N"], state["right_force_N"]) > 0.2, "Physical handle contact was lost")
        relative = [h-t for h, t in zip(state["hand_xyz"], state["tray_xyz"])]
        baseline = [h-t for h, t in zip(grasp["hand_xyz"], grasp["tray_xyz"])]
        slip = distance(relative, baseline)
        require(slip < 0.003, "Tray slip exceeded 3 mm")
    if stage == "transfer":
        require(state["tray_xyz"][2]-reference["tray_xyz"][2] > plan["lift_height_m"]-0.005,
                "Insufficient clearance during transfer")
        require(state["support_force_N"]+state["bin_floor_force_N"][index] < 0.1,
                "Tray must be airborne during transfer")
    if stage == "lift" and state["tray_xyz"][2]-reference["tray_xyz"][2] > 0.005:
        require(state["support_force_N"] < 0.1, "Lifted tray still has source support force")
    if stage in {"lower", "release", "retreat", "return", "home", "verify"}:
        require(state["support_force_N"] < 0.1, "Tray remained on its initial support")
        require(footprint_inside(state, plan), "Tray footprint is outside the selected bin")
        require(state["bin_floor_force_N"][index] < 2*weight, "Bin-floor force exceeded twice tray weight")
    if stage in {"release", "retreat", "return", "home", "verify"}:
        require(distance(state["tray_xyz"], target) < 0.003, "Placed tray moved from the bin centre")
        require(state["bin_floor_force_N"][index] > 0.5, "Placed tray lost its bin-floor support")
    if stage in {"retreat", "return", "home", "verify"}:
        require(state["left_force_N"]+state["right_force_N"] < 0.1, "Finger contact persisted after release")
    if stage == "verify":
        require(abs(state["bin_floor_force_N"][index]-weight) < 0.5,
                "Released tray weight is not supported by bin floor")
        require(max(abs(v) for v in state["tray_linear_velocity_m_s"]) < 0.002, "Released tray did not settle")
        require(max(abs(v) for v in state["tray_angular_velocity_rad_s"]) < 0.02, "Released tray rotates after release")
    return slip
