"""Release regression with bounded estimated-height bias; truth only grades."""
import copy,json
from pathlib import Path
import yaml
from native_camera_pickup import run
from native_sort import physical_sort

def run_probe(library,description,perception,control,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    settings=yaml.safe_load((control/'config/m3d.yaml').read_text());records={}
    cases=[('old_release',.0022,'bin_a','red_clear',0.)]
    cases += [(f'{name}-{bias:+.4f}-wait-{wait:.2f}',bias,bin_id,fixture,wait) for name,bin_id,fixture in
              [('red','bin_a','red_clear'),('green','bin_b','green_clear')]
              for bias in (-.003,0.,.0022,.003) for wait in (0.,1.05,1.75)]
    for name,bias,bin_id,fixture,wait in cases:
        path=folder/name;path.mkdir(exist_ok=True);config=copy.deepcopy(settings)
        config['native_release_action_wait_s']=wait
        if name=='old_release':config.update(release_root_z_m=.3075,lower_duration_s=5.,release_unload_positions_m=[],use_wrist_support=False)
        def after(plant,reference,grasp):
            estimated=copy.deepcopy(reference);estimated['camera_root_xyz'][2]+=bias
            return physical_sort(plant,estimated,grasp,config,dict(bin_id=bin_id),path)
        try:
            run(library,description/f'mujoco/static_grasp/scene_camera_m3d_{fixture}.xml',perception,control,path,
                after_pickup=after,preserve_hold_reference=True,wrist_support=True)
            result=dict(status='PASS')
        except RuntimeError as error:result=dict(status='FAIL',error=str(error))
        trace_path=path/'sort-physical-trace.json'
        trace=json.loads(trace_path.read_text()) if trace_path.exists() else []
        result.update(height_bias_m=bias,bin_id=bin_id,action_wait_s=wait,
            maximum_floor_force_N=max((max(row['bin_floor_force_N']+row.get('bin_floor_step_peak_N',[])) for row in trace),default=None))
        records[name]=result;print('Release regression '+name+': '+result['status'],flush=True)
    negative=records['old_release']['status']=='FAIL' and 'Bin floor impact exceeds bound' in records['old_release'].get('error','')
    positive=all(v['status']=='PASS' for name,v in records.items() if name!='old_release')
    result=dict(status='PASS' if negative and positive else 'FAIL',records=records,
        scope='Native free-payload physics, both bins, estimated-height offsets, gripper dwell and every-physics-step floor peaks; live ROS middleware NOT_RUN')
    (folder/'release-regression.json').write_text(json.dumps(result,indent=2)+'\n')
    assert negative,'Old height-bias pressure failure was not reproduced'
    assert positive,'Controlled release failed under bounded height bias'
    return result
