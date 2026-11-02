#!/usr/bin/env python3
"""Train a small RGB network, select on validation and scale on calibration."""
import argparse,json
from pathlib import Path
import numpy as np,yaml
from ament_index_python.packages import get_package_share_directory
from classifier_core import Network,fit_temperature,nll,metrics,digest,config_digest,require
from m3c_data import audit_splits

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);a=p.parse_args()
    c=yaml.safe_load((Path(get_package_share_directory('sorting_perception'))/'config/m3c.yaml').read_text());d=json.loads((a.dataset_dir/'dataset.json').read_text());audit=audit_splits(d['rows'],c)
    require(d['config_sha256']==config_digest(c),'dataset_config_changed')
    with np.load(a.dataset_dir/'features.npz',allow_pickle=False) as f:x=f['x'];y=f['y']
    require(len(d['rows'])==len(y),'dataset_row_mismatch');split=np.asarray([r['split'] for r in d['rows']])
    net=Network.initial(c);selection=net.train(x[split=='train'],y[split=='train'],(x[split=='validation'],y[split=='validation']),c)
    calibration=net.logits(x[split=='calibration']);t=fit_temperature(calibration,y[split=='calibration'],c['temperature_bounds'])
    a.model_dir.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.model_dir/'weights.npz',w1=net.w1,b1=net.b1,w2=net.w2,b2=net.b2)
    meta=dict(class_ids=c['class_ids'],temperature=t,config_sha256=config_digest(c),weights_sha256=digest(a.model_dir/'weights.npz'),
        dataset_manifest_sha256=digest(a.dataset_dir/'dataset.json'),features_sha256=digest(a.dataset_dir/'features.npz'),split_audit=audit,
        selection=selection,calibration_nll_before=nll(calibration,y[split=='calibration']),calibration_nll_after=nll(calibration,y[split=='calibration'],t),
        test_used_for_training=False,test_used_for_calibration=False,network=f"{3*c['input_size']**2} RGB inputs, {c['hidden_units']} ReLU units, {len(c['class_ids'])} logits; NumPy Adam",scope='Synthetic fruit skin classification')
    (a.model_dir/'model.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(status='PASS',temperature=t,selection=selection,model_directory=str(a.model_dir)),indent=2))
if __name__=='__main__':main()
