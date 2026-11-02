"""Fast capture-time and running-configuration checks before physical trials."""
from types import SimpleNamespace
from unittest.mock import Mock
from geometry_msgs.msg import TransformStamped,WrenchStamped
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType
from tf2_ros import Buffer
from wrist_support import WristObserver
from m2b_common import timestamp
from ownership_core import require

SMOOTHER='sorting_control/PrecisionAccelerationLimitedPlugin'

def capture_pairing_check(settings):
    records={}
    for name,tf_period in [('negative_20_Hz',50_000_000),('configured_100_Hz',10_000_000)]:
        paired=0;missed=0;ages=[]
        for now in range(1_064_000_000,1_265_000_000,20_000_000):
            # An adapter supplies the real TF buffer but creates no ROS node.
            adapter=SimpleNamespace(node=Mock(),buffer=Buffer(),s=settings,now_ns=lambda:now)
            observer=WristObserver(adapter)
            for stamp in range(1_000_000_000,now+1,tf_period):
                if stamp+12_000_000>now:continue
                tf=TransformStamped();tf.header.frame_id='world';tf.child_frame_id='panda_hand';tf.header.stamp=timestamp(stamp);tf.transform.rotation.w=1.
                adapter.buffer.set_transform(tf,'component_test')
            for stamp in range(1_000_000_000,now+1,10_000_000):
                if stamp+8_000_000>now:continue
                force=WrenchStamped();force.header.frame_id='panda_hand';force.header.stamp=timestamp(stamp);force.wrench.force.z=-12.5568;observer.receive(force)
            try:stamp,value=observer.sample();paired+=1;ages.append((now-stamp)/1e9)
            except RuntimeError:missed+=1
        records[name]=dict(paired=paired,unavailable=missed,maximum_age_s=max(ages,default=None))
    require(records['negative_20_Hz']['unavailable']>0,'Capture timing negative case did not reproduce')
    require(records['configured_100_Hz']['unavailable']==0,'Configured capture-time TF pairing failed')
    return dict(status='PASS',scope='Real tf2 buffer under controlled transport delays; live scheduling remains unmeasured',cases=records)

def runtime_configuration(e):
    expected={
      '/robot_state_publisher':{'use_sim_time':True,'publish_frequency':100.},
      '/sorting_servo':{'use_sim_time':True,'robot_description_kinematics.panda_arm.epsilon':1e-7,
          'moveit_servo.use_smoothing':True,'moveit_servo.smoothing_filter_plugin_name':SMOOTHER,'moveit_servo.publish_period':.01},
      '/controller_manager':{'use_sim_time':True,'update_rate':100}}
    observed={}
    for name,values in expected.items():
        client=e.node.create_client(GetParameters,name+'/get_parameters')
        try:
            e.until(client.service_is_ready,5.)
            response=e.future(client.call_async(GetParameters.Request(names=list(values))),5.)
            require(len(response.values)==len(values),'Incomplete parameter response from '+name)
            observed[name]={}
            for (key,wanted),value in zip(values.items(),response.values):
                if isinstance(wanted,bool):valid=value.type==ParameterType.PARAMETER_BOOL;actual=value.bool_value
                elif isinstance(wanted,str):valid=value.type==ParameterType.PARAMETER_STRING;actual=value.string_value
                elif isinstance(wanted,int):valid=value.type==ParameterType.PARAMETER_INTEGER;actual=value.integer_value
                else:valid=value.type==ParameterType.PARAMETER_DOUBLE;actual=value.double_value
                observed[name][key]=actual
                require(valid and (actual==wanted if not isinstance(wanted,float) else abs(actual-wanted)<1e-12),
                    'Running parameter mismatch: '+name+' '+key+' expected '+str(wanted)+' received '+str(actual))
        finally:e.node.destroy_client(client)
    return dict(status='PASS',parameters=observed)
