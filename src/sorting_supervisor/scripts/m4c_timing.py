"""Timing allocations and stationary admission. Captures keep their source stamps."""
import math
import gc
import sys
import resource
import time
from copy import deepcopy
from collections import Counter,deque
from classifier_core import require
from m4c_upstream import UpstreamTrace


def timing_budget(c):
    period=c['capture_period_budget_s']
    require(0<1./c['camera_rate_hz']<=period,'Camera period exceeds its timing allocation')
    require(c['missing_capture_reserve']==1,'M4C timing checks reserve one missing capture')
    gap=2*period+c['capture_gap_jitter_s']
    age=gap+c['delivery_budget_s']+c['callback_budget_s']
    require(age+.002<c['maximum_capture_age_s'],'A missing capture exceeds the active age allocation')
    require(c['maximum_pair_render_p95_wall_s']<period,'Rendering exceeds the capture-period allocation')
    return dict(capture_period_budget_s=period,maximum_capture_gap_s=gap,
                delivery_budget_s=c['delivery_budget_s'],callback_budget_s=c['callback_budget_s'],
                missing_capture_reserve=1,allocated_capture_age_s=age,
                age_with_guard_tick_s=age+.002,active_age_limit_s=c['maximum_capture_age_s'])


class CaptureTiming:
    def __init__(self,c,begin_ns):
        self.c=c;self.budget=timing_budget(c);self.begin_ns=begin_ns
        self.pending=deque();self.recent=deque(maxlen=8)
        self.identity=None;self.last_capture=None;self.frames=0
        self.maximum_delivery_s=0.;self.maximum_gap_s=0.;self.failure=None
        self.status='MEASURING';self.admission=None
        self.first_violation=None;self.first_delivery_violation=None
        self.worst_delivery_sample=None;self.worst_capture_gap_sample=None
        self.violation_counts=Counter();self.upstream=UpstreamTrace()

    def note(self,b,publication_ns,receipt_clock_ns,*,context=None):
        if b['capture_ns']<=self.begin_ns:return
        self.pending.append(dict(capture_ns=b['capture_ns'],publication_ns=publication_ns,
            receipt_clock_ns=receipt_clock_ns,receipt_wall_s=time.monotonic(),
            receipt_context=deepcopy(context),episode_id=b['episode_id'],track_id=b['track_id'],valid=b['valid']))
        if len(self.pending)>128:
            self.failure='Target timing queue exceeded its bound'
            while len(self.pending)>128:self.pending.popleft()

    def sample(self,now_ns,*,enforce=True,context=None,batch=None,pipeline_status=None):
        for row in list(self.pending):
            self.pending.popleft();at=row['capture_ns'];identity=(row['episode_id'],row['track_id'])
            age=(now_ns-at)/1e9
            gap=(at-self.last_capture)/1e9 if self.last_capture is not None else None
            violations=[]
            if not row['valid']:violations.append('invalid_target')
            if age<0:violations.append('future_capture')
            if age>self.c['delivery_budget_s']:violations.append('delivery_budget_exceeded')
            if self.identity is not None and identity!=self.identity:violations.append('track_identity_changed')
            if gap is not None and (gap<=0 or gap>self.budget['maximum_capture_gap_s']):
                violations.append('capture_gap_exceeded')
            self.identity=identity;self.last_capture=at;self.frames+=1
            detail=dict(row,batch_clock_ns=now_ns,delivery_age_s=age,capture_gap_s=gap,violations=violations,
                capture_to_publication_sim_s=(row['publication_ns']-at)/1e9,
                publication_to_receipt_clock_delta_s=(row['receipt_clock_ns']-row['publication_ns'])/1e9,
                receipt_to_check_clock_delta_s=(now_ns-row['receipt_clock_ns'])/1e9,
                receipt_to_check_wall_s=time.monotonic()-row['receipt_wall_s'])
            self.violation_counts.update(violations)
            worst_age=self.worst_delivery_sample is None or age>self.maximum_delivery_s
            worst_gap=gap is not None and (self.worst_capture_gap_sample is None or gap>self.maximum_gap_s)
            first=bool(violations) and self.first_violation is None
            first_delivery='delivery_budget_exceeded' in violations and self.first_delivery_violation is None
            if worst_age or worst_gap or first or first_delivery:
                retained=deepcopy(dict(detail,check_context=context,callback_batch=batch,pipeline_status=pipeline_status))
                if worst_age:self.worst_delivery_sample=retained
                if worst_gap:self.worst_capture_gap_sample=retained
                if first:self.first_violation=retained
                if first_delivery:self.first_delivery_violation=retained
                self.upstream.pin([self.first_violation,self.first_delivery_violation,self.worst_delivery_sample,self.worst_capture_gap_sample])
            self.maximum_delivery_s=max(self.maximum_delivery_s,age)
            if gap is not None:self.maximum_gap_s=max(self.maximum_gap_s,gap)
            self.recent.append(detail)
            if violations and self.failure is None:self.failure=', '.join(violations)
        if enforce:require(self.failure is None,'Stationary timing admission failed: '+str(self.failure))

    def ready(self,now_ns,**diagnostics):
        self.sample(now_ns,**diagnostics)
        if (now_ns-self.begin_ns)/1e9<self.c['timing_probe_sim_s']:return False
        minimum=math.ceil(.9*self.c['timing_probe_sim_s']/self.c['capture_period_budget_s'])
        require(self.frames>=minimum,'Too few target updates during stationary timing admission')
        self.status='PASS';self.admission=self.report();return True

    def report(self):
        return dict(status='FAIL' if self.failure else self.status,frames=self.frames,
            maximum_delivery_age_s=self.maximum_delivery_s,maximum_capture_gap_s=self.maximum_gap_s,
            failure=self.failure,budget=self.budget,recent=list(self.recent),
            violation_counts=dict(self.violation_counts),first_violation=self.upstream.decorate(self.first_violation),
            first_delivery_violation=self.upstream.decorate(self.first_delivery_violation),
            worst_delivery_sample=self.upstream.decorate(self.worst_delivery_sample),
            worst_capture_gap_sample=self.upstream.decorate(self.worst_capture_gap_sample),upstream_diagnostic=self.upstream.report(),
            scope='Observed target delivery; allocation violations after admission are diagnostic. Active age guard remains strict.')


class CollectionProbe:
    """Observe cyclic GC. Do not enable, disable, collect or change its thresholds."""
    def __init__(self,context):
        self.context=context;self.active={};self.recent=deque(maxlen=8)
        self.counts=Counter();self.worst=None;self.sequence=0

    def start(self):
        if self.callback not in gc.callbacks:gc.callbacks.append(self.callback)

    def close(self):
        if self.callback in gc.callbacks:gc.callbacks.remove(self.callback)

    def callback(self,phase,info):
        generation=info['generation']
        if phase=='start':
            self.active[generation]=dict(start_wall_s=time.monotonic(),start_cpu_s=time.thread_time(),context=self.context())
        elif phase=='stop' and generation in self.active:
            row=self.active.pop(generation);self.sequence+=1
            row.update(sequence=self.sequence,generation=generation,end_wall_s=time.monotonic(),
                       end_cpu_s=time.thread_time(),collected=info.get('collected'),uncollectable=info.get('uncollectable'))
            row['wall_s']=row['end_wall_s']-row['start_wall_s'];row['cpu_s']=row['end_cpu_s']-row['start_cpu_s']
            self.counts[generation]+=1;self.recent.append(row)
            if self.worst is None or row['wall_s']>self.worst['wall_s']:self.worst=row

    def overlaps(self,start,end):
        if start is None:return []
        rows={r['sequence']:r for r in self.recent}
        if self.worst is not None:rows[self.worst['sequence']]=self.worst
        return [r for r in rows.values() if r['start_wall_s']<=end and r['end_wall_s']>=start]

    def report(self):
        return dict(collections_by_generation=dict(self.counts),maximum_collection=self.worst,recent=list(self.recent),
                    scope='Observer only. CPU time is for the current thread; wall time can include scheduling or blocking.')


class FullCollectionWindow:
    """Defer automatic generation-2 scans for one bounded Python 3.12 test."""
    def __init__(self):
        self.record=dict(started=False,restored=False,applied=False)

    def begin(self):
        require(sys.version_info[:2]==(3,12),'Full-collection window requires the validated Python 3.12 runtime')
        require(not self.record['started'],'Full-collection window already started')
        self.saved=gc.get_threshold();self.enabled=gc.isenabled()
        active=(self.saved[0],self.saved[1],max(self.saved[2],2**31-1))
        if self.enabled and self.saved[0]!=0:
            gc.set_threshold(*active);self.record['applied']=True
        self.before=gc.get_stats()[2]['collections']
        self.record.update(started=True,original_thresholds=list(self.saved),window_thresholds=list(active),
            automatic_collection_enabled=self.enabled,start_wall_s=time.monotonic(),
            peak_rss_before_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def restore(self,reason):
        if not self.record['started'] or self.record['restored']:return
        self.record.update(active_full_collections=gc.get_stats()[2]['collections']-self.before,
            restored=True,restore_reason=reason,end_wall_s=time.monotonic(),
            peak_rss_after_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if self.record['applied']:gc.set_threshold(*self.saved)
        self.record['restored_thresholds']=list(gc.get_threshold())

    def report(self):
        return dict(self.record,scope='One bounded executor/evaluator process. Younger collections remain enabled; explicit full collections remain possible.')
