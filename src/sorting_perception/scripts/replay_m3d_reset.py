#!/usr/bin/env python3
"""Replay saved contact/phase data and the reset-entry callback; no ROS node."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import yaml
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
from m3d_evaluator import SortEvaluator
from ownership_core import require

def observer():
    e=SortEvaluator.__new__(SortEvaluator)
    control=Path(get_package_share_directory('sorting_control'))
    e.s=yaml.safe_load((control/'config/m3d.yaml').read_text())
    e.c=yaml.safe_load((control/'config/m3b.yaml').read_text())
    e.phase='startup';e.control_state={};e.reset_in_progress=False;e.guarded=True
    e.grasp=None;e.finger_start=None;e.contact={};e.history=[];e.lifted_samples=[];e.grip_trace=[]
    e.maximum_slip=0.;e.maximum_world_offset_change=0.;e.maximum_rise=0.;e.maximum_transfer_rise=0.
    e.placement_started=False;e.release_seen=False;e.last_placement_phase=None
    e.decision=None;e.node=Mock();e.scene_clients=[]
    return e

def state(e,phase):e.receive_control(String(data=json.dumps(dict(phase=phase,active=phase!='idle'))))

def reset_entry(e,sample):
    """Exercise the real reset methods up to the first reset-guard request."""
    class GuardReached(Exception):pass
    e.owner=dict(owner='HOLD',busy=False,ready=True)
    e.reset_guard=object()
    def spin(node,timeout_sec):
        # Probe.until spins before it tests its predicate, even if ready already.
        state(e,'idle');e.receive_contact(String(data=json.dumps(sample)))
    def call(client,request):
        require(client is e.reset_guard and request.data,'Unexpected reset operation')
        raise GuardReached()
    e.call=call
    try:
        with patch('test_arm_ownership.rclpy.spin_once',side_effect=spin):e.reset_episode()
    except GuardReached:
        return dict(status='PASS',reached='reset_guard_request',guarded=e.guarded,
                    phase=e.phase,last_placement_phase=e.last_placement_phase)
    raise RuntimeError('Reset guard request was not reached')

def replay(trial_dir):
    folder=Path(trial_dir);policy=json.loads((folder/'controller.json').read_text())
    require(policy.get('status')=='PASS' and policy.get('action')=='PICK_PLACE','Replay needs a completed placement controller record')
    trace=json.loads((folder/'physical-trace.json').read_text());require(bool(trace),'Physical trace is empty')
    events=sorted(policy['events'],key=lambda x:x['time_ns']);e=observer()
    e.decision=SimpleNamespace(proposed_bin=policy['class_decision']['bin_id'])
    position=0;replayed=0
    for sample in trace:
        while position<len(events) and events[position]['time_ns']<=round(sample['sim_time']*1e9):
            state(e,events[position]['phase']);position+=1
        e.receive_contact(String(data=json.dumps(sample)));replayed+=1
    require(e.last_placement_phase=='verify_placement','Replay did not reach completed placement')
    require(e.grasp is not None,'Replay did not establish a physical grip baseline')
    held_slip=e.maximum_slip
    # Use the recorded final, released state for a callback during reset entry.
    result=reset_entry(e,trace[-1])
    require(result['guarded'],'Physical guards were disabled before reset authorization')
    return dict(status='PASS',scope='Saved contact/phase replay through reset guard request; no live services, world reset or robot motion',
                contact_samples=replayed,maximum_held_slip_m=held_slip,reset_entry=result,
                final_recorded_bin_floor_force_N=trace[-1]['bin_floor_force_N'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=replay(a.trial_dir)
    except Exception as error:report=dict(status='FAIL',error=str(error),scope='Offline replay only')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
