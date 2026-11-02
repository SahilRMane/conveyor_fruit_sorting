"""Truth belongs to this evaluator, never to the class/bin policy."""
import copy,json,math
from pathlib import Path
import numpy as np,yaml
from ament_index_python.packages import get_package_share_directory
from m3b_evaluator import Evaluator
from sorting_interfaces.msg import Decision
from grasp_core import grasp_slip_metrics,require_grasp_slip
from ownership_core import require
from camera_scene_checks import collision_allowed
from camera_scene import TRAY_IDS
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents

class SortEvaluator(Evaluator):
    def __init__(self,node):
        self.s=yaml.safe_load((Path(get_package_share_directory('sorting_control'))/'config/m3d.yaml').read_text());self.decision=None;self.placement_started=False;self.release_seen=False;self.maximum_transfer_rise=0.;self.audit_diagnostic=None;self.last_placement_phase=None
        super().__init__(node);node.create_subscription(Decision,'/sorting/decisions',lambda m:setattr(self,'decision',m),10)
    def prepare_trial(self):
        super().prepare_trial();self.decision=None;self.placement_started=False;self.release_seen=False;self.maximum_transfer_rise=0.;self.last_placement_phase=None
    def receive_control(self,m):
        super().receive_control(m)
        if self.reset_in_progress:return
        if self.phase in {'transfer_retract','transfer_raise','transfer','lower','release','retreat','verify_placement'}:
            self.last_placement_phase=self.phase
        elif self.phase=='idle' and self.last_placement_phase is not None:
            # State and result use separate topics. Idle may arrive first.
            # Continue checking the actual last phase while the result arrives.
            self.phase=self.last_placement_phase
    def check(self,value):
        if self.phase.startswith('transfer'):self.placement_started=True
        if self.phase=='release':self.release_seen=True
        if self.phase=='idle' and self.last_placement_phase is not None:self.phase=self.last_placement_phase
        if self.phase not in {'transfer_retract','transfer_raise','transfer','lower','release','retreat','verify_placement'}:
            return super().check(value)
        require(value['forbidden_contacts']==0 and value['tray_floor_force_N']<.1 and sum(value['bin_wall_force_N'])<.1,'Forbidden placement contact')
        require(max(value['left_force_N'],value['right_force_N'])<15.,'Placement finger force exceeds 15 N')
        require(abs(value['external_force_x_N'])<1e-6,'Conveyor force must stop before transfer')
        require(self.grasp is not None,'Transfer has no stable physical grip baseline')
        rise=value['tray_xyz'][2]-self.grasp['tray_xyz'][2];self.maximum_transfer_rise=max(self.maximum_transfer_rise,rise);require(-.003<=rise<self.s['maximum_transfer_rise_m'],'Transfer height outside bound')
        require(abs(np.dot(value['tray_quat_wxyz'],self.grasp['tray_quat_wxyz']))>math.cos(self.s['maximum_tilt_rad']/2),'Placement tray tilt exceeds 3 degrees')
        if self.phase in {'transfer_retract','transfer_raise','transfer','lower'}:
            metrics=grasp_slip_metrics(value,self.grasp);require_grasp_slip(metrics,.003);self.maximum_slip=max(self.maximum_slip,metrics['relative_slip_m']);self.grip_trace.append(dict(metrics,phase=self.phase,sim_time=value['sim_time']))
            require(min(value['left_force_N'],value['right_force_N'])>.2,'Physical contact lost during placement')
        if self.phase=='transfer':require(rise>self.s['minimum_airborne_transfer_rise_m'] and value['support_force_N']+sum(value['bin_floor_force_N'])<.1,'Transfer clearance/support condition failed')
        if self.decision and self.decision.proposed_bin in self.s['bins']:
            index=self.s['bins'][self.decision.proposed_bin]['index'];require(value['bin_floor_force_N'][1-index]<.1,'Tray touched the unselected bin')
            require(value['bin_floor_force_N'][index]<2*.52*9.81,'Bin-floor force exceeded twice payload weight')
    def audit(self):
        # Names and subscription endpoints become visible independently. Wait
        # only for missing discovery data; an observed violation fails at once.
        expected={
            'handle_localizer':{'/camera/wrist/image_raw','/camera/wrist/camera_info','/sorting/episode'},
            'camera_motion_filter':{'/sorting/perception/wrist/handle_observation','/sorting/episode'},
            'camera_grasp_scene':{'/sorting/targets','/joint_states','/sorting/episode'},
            'fruit_classifier':{'/camera/wrist/image_raw','/sorting/targets','/sorting/episode'},
            'camera_sort_controller':{'/sorting/classified_targets','/sorting/perception/class_observation','/sorting/targets','/sorting/arm/owner_state'}}
        truth_topics=('/sorting/contact_state','/sorting/ground_truth/camera_sample','/sorting/reset_state')
        subscriptions={}
        def ready():
            graph=self.node.get_node_names_and_namespaces();root_names={n for n,space in graph if space=='/'}
            missing=sorted(set(expected)-root_names);oracles=sorted(n for n,_ in graph if 'oracle' in n)
            subscriptions.clear();missing_inputs={};violations=[];missing_observers=[]
            for name,required in expected.items():
                if name not in root_names:continue
                topics=sorted(t for t,_ in self.node.get_subscriber_names_and_types_by_node(name,'/'));subscriptions[name]=topics
                absent=sorted(required-set(topics))
                if absent:missing_inputs[name]=absent
                bad=[t for t in topics if t.startswith('/sorting/ground_truth/') or t.startswith('/sorting/debug/') or t in ('/sorting/contact_state','/sorting/reset_state')]
                if bad:violations.append(dict(node=name,topics=bad))
            for topic in truth_topics:
                observers=self.node.get_subscriptions_info_by_topic(topic)
                if not any(p.node_name=='test_m3d' and p.node_namespace=='/' for p in observers):missing_observers.append(topic)
                for p in observers:
                    if p.node_name!='test_m3d' or p.node_namespace!='/':violations.append(dict(topic=topic,node=p.node_name,namespace=p.node_namespace))
            self.audit_diagnostic=dict(nodes=sorted(space.rstrip('/')+'/'+name for name,space in graph),
                missing_nodes=missing,missing_policy_subscriptions=missing_inputs,
                missing_evaluator_subscriptions=missing_observers,oracle_nodes=oracles,violations=violations)
            require(not oracles,'Oracle node present in classified camera pipeline: '+', '.join(oracles))
            require(not violations,'Truth input or subscriber outside M3D evaluator: '+json.dumps(violations))
            return not (missing or missing_inputs or missing_observers)
        try:self.until(ready,10.)
        except TimeoutError as error:
            raise TimeoutError('Classified camera pipeline discovery timed out; see startup_audit_diagnostic: '+json.dumps(self.audit_diagnostic)) from error
        return dict(status='PASS',policy_subscriptions=copy.deepcopy(subscriptions),discovery=copy.deepcopy(self.audit_diagnostic))
    def verify_sort(self,policy,expected_class,expect_pass=False):
        require(policy['status']=='PASS',policy.get('error','Policy failed'));decision=policy['class_decision']
        if expect_pass:
            require(policy['action']=='PASS' and decision['confidence']<self.s['minimum_confidence'],'Covered target must pass')
            initial=np.asarray(self.plan['initial_state']['tray_xyz']);final=np.asarray(self.contact['tray_xyz'])
            require(self.history and all(np.linalg.norm(np.asarray(v['tray_xyz'])[1:]-initial[1:])<.003 and v['left_force_N']+v['right_force_N']<.1 for v in self.history),'PASS lifted or touched the physical tray')
            require(.02<final[0]-initial[0]<.08,'PASS did not let the covered tray continue on the conveyor')
            require(self.grasp is None and self.maximum_rise<.003 and self.contact['support_force_N']>4. and min(self.contact['finger_positions_m'])>.039,'PASS changed the physical tray/gripper')
            return dict(status='PASS',action='PASS',physical_grasp='NOT_RUN',conveyor_displacement_m=float(final[0]-initial[0]))
        require(policy['action']=='PICK_PLACE' and decision['class_id']==expected_class and decision['bin_id']==self.s['class_to_bin'][expected_class],'Wrong RGB class or selected bin')
        require(decision['episode_id']==policy['episode_id'] and decision['track_id']==policy['motion_track_id'],'Class decision track/episode mismatch')
        # Score the pickup at its recorded completion, before later placement lowered it.
        current=self.contact;complete=policy['pickup_complete_ns'];pickup_contact=min(self.history,key=lambda x:abs(round(x['sim_time']*1e9)-complete))
        self.contact=pickup_contact
        try:pickup=super().verify_physical(policy)
        finally:self.contact=current
        index=self.s['bins'][decision['bin_id']]['index'];center=np.asarray(self.s['bins'][decision['bin_id']]['center'])
        settled=[x for x in self.history if x['sim_time']>current['sim_time']-self.s['placement_settle_s']]
        require(len(settled)>=50,'Insufficient settled placement samples')
        for value in settled:
            from prediction_core import rotation
            from sort_core import footprint_inside_estimate
            q=value['tray_quat_wxyz'];world=np.eye(4);world[:3,:3]=rotation([q[1],q[2],q[3],q[0]]);world[:3,3]=value['tray_xyz']
            require(footprint_inside_estimate(world,decision['bin_id'],self.s),'Physical footprint is outside the bin')
            require(np.linalg.norm(np.asarray(value['tray_xyz'])-center)<.003,'Physical tray is outside bin-centre tolerance')
            require(value['bin_floor_force_N'][index]>.5 and abs(value['bin_floor_force_N'][index]-.52*9.81)<.5 and value['bin_floor_force_N'][1-index]<.1,'Wrong or incomplete bin support')
            require(value['left_force_N']+value['right_force_N']<.1,'Fingers retained physical contact')
            require(max(abs(v) for v in value['tray_linear_velocity_m_s'])<.002 and max(abs(v) for v in value['tray_angular_velocity_rad_s'])<.02,'Placed tray not settled')
        require(self.owner.get('owner')=='HOLD' and self.owner['stats']['overlap_violations']==0,'Final owner/overlap failure')
        return dict(status='PASS',action='PICK_PLACE',true_class=expected_class,selected_bin=decision['bin_id'],pickup=pickup,placement_samples=len(settled),final_state=current,maximum_hand_frame_slip_m=self.maximum_slip,maximum_transfer_rise_m=self.maximum_transfer_rise)
    def reset_episode(self):
        # The base method spins while it waits for reset authorization. Keep
        # the placement phase and its physical guards until that reset ends.
        result=super().reset_episode()
        self.last_placement_phase=None;self.placement_started=False;self.release_seen=False
        for client in self.scene_clients:
            request=GetPlanningScene.Request();request.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            matrix=self.call(client,request).scene.allowed_collision_matrix
            require(not any(collision_allowed(matrix,part,bin_id+'_floor') for part in TRAY_IDS for bin_id in ('bin_a','bin_b')),'Destination permissions remained after reset')
        result['destination_permissions_reset']='PASS';return result
