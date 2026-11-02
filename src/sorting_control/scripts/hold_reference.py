"""Bounded position reference for a hold; uses controller/encoder data only."""
import math
from ownership_core import NAMES,positions_valid,require

def select_hold(measured,sample,now_ns,wall_now,preserve=False,maximum_delta=.05,maximum_age=.15,wall_timeout=.75):
    require(positions_valid(measured),'Invalid measured hold positions')
    if not preserve:return list(measured),dict(policy='measured',maximum_delta_rad=0.)
    require(sample is not None,'Arm-controller reference is not available')
    require(-.05<=(now_ns-sample['stamp_ns'])/1e9<maximum_age and 0<=wall_now-sample['received_wall']<wall_timeout,'Arm-controller reference is stale')
    names=sample['names'];values=sample['positions']
    require(len(names)==7 and len(set(names))==7 and set(names)==set(NAMES) and len(values)==7,'Arm-controller reference joint set is incomplete')
    ordered=[float(values[names.index(n)]) for n in NAMES]
    require(positions_valid(ordered),'Invalid arm-controller reference positions')
    delta=max(abs(a-b) for a,b in zip(ordered,measured));require(math.isfinite(delta) and delta<maximum_delta,'Arm-controller reference differs too far from feedback')
    return ordered,dict(policy='controller_reference',maximum_delta_rad=delta,reference_stamp_ns=sample['stamp_ns'],reference_age_sim_s=(now_ns-sample['stamp_ns'])/1e9)
