"""Native reproduction of the recorded loaded-hold drift; no ROS node."""
import json,numpy as np
from native_camera_pickup import run

def run_probe(library,scene,perception,control,folder):
    folder.mkdir(parents=True,exist_ok=True);records={}
    for mode in ('measured','controller_reference'):
        path=folder/mode;path.mkdir(exist_ok=True)
        def check(plant,reference,grasp):
            # Reach the hand pose recorded immediately before the live first hold.
            # Move position actuators only. Keep the tray physically free.
            goal=np.array([.68046,-.00015,.57575]);plant.call('nc_resume')
            for _ in range(700):plant.control(goal,[0,0,0],0.)
            first=plant.truth();rows=[]
            for index in range(3):
                plant.call('nc_hold' if mode=='measured' else 'nc_hold_reference')
                for _ in range(80):
                    plant.call('nc_step',.01,0.);state=plant.truth()
                    tilt=float(2*np.arccos(np.clip(abs(np.dot(state['tray_quat_wxyz'],grasp['tray_quat_wxyz'])),0,1)))
                    rows.append(dict(hold=index+1,tilt_rad=tilt,**state))
            value=dict(maximum_tilt_deg=float(np.rad2deg(max(s['tilt_rad'] for s in rows))),hand_drop_m=float(first['hand_xyz'][2]-rows[-1]['hand_xyz'][2]),final_state=rows[-1])
            (path/'hold-physical-trace.json').write_text(json.dumps(rows,indent=2)+'\n');records[mode]=value;return value
        run(library,scene,perception,control,path,after_pickup=check,preserve_hold_reference=True)
    assert records['measured']['maximum_tilt_deg']>3.,'Recorded hold drift was not reproduced'
    assert records['controller_reference']['maximum_tilt_deg']<3. and abs(records['controller_reference']['hand_drop_m'])<.001,'Reference-preserving hold did not maintain the loaded pose'
    report=dict(status='PASS',records=records,scope='Native loaded-hold regression at recorded live hand pose; live ROS NOT_RUN')
    (folder/'hold-regression.json').write_text(json.dumps(report,indent=2)+'\n');return report
