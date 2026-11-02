"""Offline grouped data preparation and split audits."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np,yaml
from classifier_core import fruit_crop,features,require,config_digest
from camera_core import localize

def audit_splits(rows,c):
    groups={};appearances={}
    for r in rows:
        require(r['split'] in c['split_group_counts'],'unknown_split')
        groups.setdefault(r['group_id'],set()).add(r['split']);appearances.setdefault(r['appearance_instance'],set()).add(r['split'])
    require(all(len(x)==1 for x in groups.values()) and all(len(x)==1 for x in appearances.values()),'group_or_appearance_leakage')
    counts={s:sum(s in v for v in groups.values()) for s in c['split_group_counts']}
    require(counts==c['split_group_counts'],'split_group_counts_mismatch')
    for group in groups:
        sample=[r for r in rows if r['group_id']==group]
        require(len(sample)==6 and {r['label'] for r in sample}=={0,1},'incomplete_counterfactual_group')
    return dict(group_counts=counts,groups_disjoint=True,appearance_instances_disjoint=True,paired_classes_per_group=True)
def prepare(raw,output,c,localization):
    data=yaml.safe_load((raw/'frames.yaml').read_text());rows=data['frames'];audit=audit_splits(rows,c);output.mkdir(parents=True,exist_ok=True)
    xs=[];labels=[];saved=[];hashes=hashlib.sha256();conditions={};errors=[]
    for row in rows:
        content=(raw/row['image']).read_bytes();hashes.update(content);rgb=cv2.cvtColor(cv2.imread(str(raw/row['image'])),cv2.COLOR_BGR2RGB)
        f=row['height']/2/np.tan(np.deg2rad(row['fovy_deg'])/2);k=np.array([[f,0,row['width']/2],[0,f,row['height']/2],[0,0,1.]])
        rotation=np.asarray(row['camera_optical_rotation']).reshape(3,3)
        found=localize(rgb,k,row['camera_xyz'],rotation,localization)
        error=float(np.linalg.norm(np.asarray(found['position'])[:2]-np.asarray(row['evaluator_handle_xyz'])[:2]));errors.append(error)
        require(error<localization['maximum_planar_error_m'],'M3C marker localization exceeds 2 mm')
        crop,bbox=fruit_crop(rgb,k,row['camera_xyz'],rotation,found['position'],c);x=features(crop,c)
        xs.append(x);labels.append(row['label']);saved.append(dict(row,crop_xyxy=bbox,position_source='neutral_RGB_marker'))
        conditions[(row['group_id'],row['label'],row['condition'])]=crop
        if row['group_id']<3:cv2.imwrite(str(output/(row['id']+'-crop.png')),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
    # Paired same-scene class changes isolate the class-bearing skin from marker/background.
    diffs={condition:[] for condition in ('overhead_hidden','wrist_clear','wrist_occluded')}
    for group in range(c['dataset_groups']):
        for condition in diffs:
            a=conditions[(group,0,condition)];b=conditions[(group,1,condition)]
            require(a.shape==b.shape,'class_changed_crop_geometry')
            diffs[condition].append(float(np.abs(a.astype(float)-b).mean()/255.))
    paired={k:dict(mean_rgb_difference=float(np.mean(v)),maximum_rgb_difference=max(v)) for k,v in diffs.items()}
    require(paired['wrist_clear']['mean_rgb_difference']>.005,'new_view_has_no_class_evidence')
    require(paired['overhead_hidden']['maximum_rgb_difference']<.001 and paired['wrist_occluded']['maximum_rgb_difference']<.001,'hidden_view_leaks_class_evidence')
    np.savez_compressed(output/'features.npz',x=np.asarray(xs,np.float32),y=np.asarray(labels,np.int64))
    manifest=dict(rows=saved,split_audit=audit,paired_view_audit=paired,raw_rgb_sha256=hashes.hexdigest(),config_sha256=config_digest(c),
        scope='Procedural textures, occluders and lighting; held-out instances within the same nuisance ranges',physics_preserved=data['physics_mass_and_inertia_preserved'],maximum_marker_error_m=max(errors))
    (output/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
