"""MoveIt collision permissions, including implicit and default entries."""
FINGERS=('panda_leftfinger','panda_rightfinger')

def collision_allowed(matrix,a,b):
    names=list(matrix.entry_names)
    if a in names and b in names:
        row=matrix.entry_values[names.index(a)].enabled
        return bool(row[names.index(b)])
    defaults=dict(zip(matrix.default_entry_names,matrix.default_entry_values))
    values=[defaults[name] for name in (a,b) if name in defaults]
    return bool(values) and all(values)

def reset_scene_diagnostic(scene,tray_ids):
    world={o.id for o in scene.world.collision_objects}
    attached={a.object.id for a in scene.robot_state.attached_collision_objects}
    permissions={f:collision_allowed(scene.allowed_collision_matrix,'handle_bar',f) for f in FINGERS}
    support_permissions={part:collision_allowed(scene.allowed_collision_matrix,part,"support_surface") for part in tray_ids}
    support_allowed=any(support_permissions.values())
    internal_permissions=[(a,b) for a in sorted(tray_ids) for b in sorted(tray_ids) if a<b and collision_allowed(scene.allowed_collision_matrix,a,b)]
    missing=sorted(tray_ids-world);left=sorted(tray_ids.intersection(attached))
    return dict(valid=not missing and not left and not any(permissions.values()) and not support_allowed and not internal_permissions,missing_world_objects=missing,
                remaining_attached_objects=left,handle_finger_permission=permissions,
                tray_support_permission=support_allowed,tray_support_permissions=support_permissions,remaining_internal_tray_permissions=internal_permissions,acm_entry_names=list(scene.allowed_collision_matrix.entry_names),world_object_count=len(world))
