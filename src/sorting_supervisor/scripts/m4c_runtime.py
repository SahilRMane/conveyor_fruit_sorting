"""Move the wrist while a separate overhead filter maintains the target track."""
from collections import deque
import json
import time
from pathlib import Path
import yaml
from rclpy.executors import SingleThreadedExecutor, TimeoutException, ShutdownException
from std_msgs.msg import Float64, String
from rclpy.qos import QoSProfile,ReliabilityPolicy
from sorting_interfaces.msg import TargetBeliefArray, Decision
from test_m4b import ClassInspectionExecutor
from m4a_runtime import stamp, ns
from m4c_core import remaining, pick_reserve, DeadlinePass
from classifier_core import require
from m4c_timing import CaptureTiming


class FeedbackExecutor(SingleThreadedExecutor):
    """Run a bounded batch of ready callbacks before the caller checks freshness."""
    def __init__(self, callback_budget_s):
        super().__init__()
        require(0 < callback_budget_s <= .015, 'Callback batch budget must be at most 15 ms')
        self.callback_budget_s = callback_budget_s
        self.callback_diagnostic = {}
        self.startup_clock_pending = None
        self.diagnostic_context = None

    def spin_once(self, timeout_sec=None):
        started = None; count = 0; maximum_callback = 0.; callback_wall = 0.; reason = 'queue_empty'
        maximum_callback_info = {}; alignment_started = None; alignment_callbacks = 0
        context_reader = getattr(self, 'diagnostic_context', None); batch_context = None
        started_cpu = None; callback_cpu = 0.
        def execute(handler, entity, node):
            nonlocal count, maximum_callback, maximum_callback_info, started, callback_wall, batch_context, started_cpu, callback_cpu
            begin = time.monotonic(); begin_cpu = time.thread_time(); count += 1
            if started_cpu is None:started_cpu = begin_cpu
            if batch_context is None and context_reader is not None:batch_context = dict(context_reader())
            if started is None: started = begin
            try:
                handler()
                handler.result()
            finally:
                elapsed = time.monotonic()-begin
                elapsed_cpu = time.thread_time()-begin_cpu
                callback_wall += elapsed; callback_cpu += elapsed_cpu
                if elapsed >= maximum_callback:
                    maximum_callback = elapsed
                    name = getattr(entity, 'topic_name', getattr(entity, 'srv_name', None))
                    maximum_callback_info = dict(duration_wall_s=elapsed,duration_cpu_s=elapsed_cpu,
                        node=node.get_name() if node is not None else None,
                        entity_type=type(entity).__name__, name=name,
                        callback=getattr(getattr(entity, 'callback', None), '__qualname__', None),
                        timer_period_ns=getattr(entity, 'timer_period_ns', None))
        try:
            while count < 64:
                if started is not None and time.monotonic()-started >= self.callback_budget_s:
                    reason = 'wall_budget'; break
                try:
                    handler, entity, node = self.wait_for_ready_callbacks(
                        timeout_sec=timeout_sec if started is None else 0.)
                except (TimeoutException, ShutdownException):
                    break
                if started is None: started = time.monotonic()
                execute(handler, entity, node)
            else:
                reason = 'callback_count'
            pending = getattr(self, 'startup_clock_pending', None)
            if pending is not None and pending():
                alignment_started = time.monotonic()
                for _ in range(64):
                    left = .05-(time.monotonic()-alignment_started)
                    if left <= 0 or not pending(): break
                    try:
                        handler, entity, node = self.wait_for_ready_callbacks(timeout_sec=min(.005, left))
                    except TimeoutException:
                        continue
                    except ShutdownException:
                        break
                    alignment_callbacks += 1
                    execute(handler, entity, node)
        except BaseException:
            reason = 'callback_error'; raise
        finally:
            ended = time.monotonic()
            elapsed = ended-started if started is not None else 0.
            batch_cpu = time.thread_time()-started_cpu if started_cpu is not None else 0.
            previous = self.callback_diagnostic
            end_context = dict(context_reader()) if context_reader is not None else None
            if batch_context is None:batch_context = end_context
            phase = (batch_context or {}).get('phase', 'unspecified')
            if phase != (end_context or {}).get('phase', 'unspecified'):phase = 'phase_transition'
            batch = dict(phase=phase,start_context=batch_context,end_context=end_context,callbacks=count,batch_wall_s=elapsed,stop_reason=reason,
                start_wall_s=started,end_wall_s=ended,callback_execution_wall_s=callback_wall,callback_execution_cpu_s=callback_cpu,batch_cpu_s=batch_cpu,
                outside_callback_wall_s=max(0.,elapsed-callback_wall),slowest_callback=maximum_callback_info,
                startup_clock_alignment=alignment_started is not None,budget_s=self.callback_budget_s,
                gc_overlaps=self.collection_probe.overlaps(started,ended) if getattr(self,'collection_probe',None) is not None else [])
            scoped = dict(previous.get('batches_by_phase', {}))
            if count:
                old = scoped.get(phase, {})
                scoped[phase] = dict(batches=old.get('batches', 0)+1,
                    maximum_batch=(batch if elapsed>old.get('maximum_batch_wall_s', -1.) else old['maximum_batch']),
                    maximum_batch_wall_s=max(elapsed,old.get('maximum_batch_wall_s', 0.)),
                    first_over_budget_batch=(old.get('first_over_budget_batch') or
                                             (batch if elapsed>self.callback_budget_s else None)))
            self.callback_diagnostic = dict(batches_by_phase=scoped,callbacks=count, batch_wall_s=elapsed, stop_reason=reason,
                current_batch=batch,
                maximum_batch=(batch if elapsed>previous.get('maximum_batch_wall_s',-1.) else previous.get('maximum_batch')),
                first_over_budget_batch=(previous.get('first_over_budget_batch') or (batch if elapsed>self.callback_budget_s else None)),
                maximum_batch_wall_s=max(previous.get('maximum_batch_wall_s', 0.), elapsed),
                maximum_callback_wall_s=max(previous.get('maximum_callback_wall_s', 0.), maximum_callback),
                maximum_callback=(maximum_callback_info if maximum_callback >= previous.get('maximum_callback_wall_s', 0.)
                                  else previous.get('maximum_callback', {})),
                budget_s=self.callback_budget_s, maximum_callbacks=64,
                bounded_batches=previous.get('bounded_batches', 0)+(reason != 'queue_empty'),
                startup_clock_wait_budget_s=.05, startup_clock_maximum_callbacks=64,
                startup_clock_callbacks=alignment_callbacks,
                startup_clock_wait_wall_s=time.monotonic()-alignment_started if alignment_started is not None else 0.,
                startup_clock_waits=previous.get('startup_clock_waits', 0)+(alignment_started is not None))


def snapshot(b):
    return dict(episode_id=b.episode_id,track_id=b.track_id,capture_ns=ns(b.header.stamp),
                frame_id=b.header.frame_id,
                valid=b.valid,camera_only=b.camera_only,view_id=b.view_id,
                state=list(b.state),covariance=list(b.covariance))


class MovingInspectionExecutor(ClassInspectionExecutor):
    def __init__(self, folder, report, config):
        super().__init__(folder,report)
        self.mc=config
        self.policy_config=config
        self.filter_config=yaml.safe_load((Path(folder)/'filter.yaml').read_text())
        self.costs=json.loads((Path(folder)/'costs.json').read_text())['upper_sim_s']
        self.bound_track=None;self.overhead=None;self.overhead_wall=0.
        self.history=deque(maxlen=1200)
        self.tracking_active=False;self.operation=None;self.deadline_diagnostic={}
        self.overhead_diagnostic={};self.overhead_frames=0
        self.timing=None;self.pipeline_status={};self.publisher_diagnostic={}
        self.create_subscription(TargetBeliefArray,'/sorting/m4c/targets',self.receive_targets,1)
        for topic in ('/sorting/perception/camera_status','/sorting/m4c/filter_status'):
            self.create_subscription(String,topic,lambda message,t=topic:self.receive_pipeline_status(t,message),1)
        for stage,topic in (('localizer','/sorting/perception/localizer_timing'),('filter','/sorting/m4c/filter_timing')):
            self.create_subscription(String,topic,lambda message,s=stage:self.receive_upstream_trace(s,message),
                                     QoSProfile(depth=64,reliability=ReliabilityPolicy.BEST_EFFORT))
        self.decisions=self.create_publisher(Decision,'/sorting/m4c/decisions',1)

    def receive_upstream_trace(self,stage,message):
        if self.timing is None:return
        try:row=json.loads(message.data)
        except (ValueError,TypeError):
            self.timing.upstream.counts['malformed_json']+=1;return
        self.timing.upstream.note(row,stage,self.episode)

    def receive_pipeline_status(self,topic,message):
        self.pipeline_status[topic]=dict(data=json.loads(message.data),received_clock_ns=self.now())

    def check_publishers(self):
        topics=('/clock','/joint_states','/sorting/episode','/sorting/arm/owner_state',
                '/camera/overhead/image_raw','/camera/wrist/image_raw','/sorting/m4c/targets',
                '/sorting/perception/localizer_timing','/sorting/m4c/filter_timing')
        self.publisher_diagnostic={topic:[dict(node=i.node_name,namespace=i.node_namespace)
            for i in self.get_publishers_info_by_topic(topic)] for topic in topics}
        bad={topic:items for topic,items in self.publisher_diagnostic.items() if len(items)!=1}
        require(not bad,'M4C requires one publisher per critical topic: '+json.dumps(bad))

    def prime_timing(self):
        self.prime_overhead();self.check_publishers()
        self.timing=CaptureTiming(self.mc,self.now())
        self.events.append(dict(state='STATIONARY_TIMING',enter_ns=self.now()))
        def ready():
            require(self.owner.get('owner')=='HOLD' and not self.owner.get('busy') and self.belt_speed==0.,
                    'Stationary timing admission requires HOLD and a stopped belt')
            self.target()
            return self.timing.ready(self.now(),**self.timing_sample_options())
        self.until(ready,timeout=self.mc['timing_probe_wall_s'])
        self.check_publishers()
        self.events.append(dict(state='TIMING_READY',enter_ns=self.now(),timing=self.timing.report()))

    def timing_context(self):
        event=self.events[-1] if self.events else {}
        phase=('startup' if self.timing is None else 'moving' if self.tracking_active else 'cleanup' if self.policy_started is not None
               else 'stationary_admission' if self.timing is not None and self.timing.status!='PASS' else 'pre_transport')
        return dict(phase=phase,state=event.get('state'),pose=event.get('pose'),
                    operation=self.operation.get('name') if self.operation else None,belt_speed_m_s=self.belt_speed)

    def timing_sample_options(self):
        return dict(context=self.timing_context(),batch=getattr(self.executor,'callback_diagnostic',{}).get('current_batch'),
                    pipeline_status=getattr(self,'pipeline_status',{}))

    def timing_diagnostic(self):
        if self.timing is not None:self.timing.sample(self.now(),enforce=False,**self.timing_sample_options())
        return dict(admission=self.timing.admission if self.timing is not None else None,
                    observed=self.timing.report() if self.timing is not None else None,
                    publishers=self.publisher_diagnostic,pipeline_status=self.pipeline_status)

    def startup_clock_pending(self):
        return (self.feedback_armed and not self.tracking_active and self.policy_started is None
                and self.failure is None and self.owner.get('owner') == 'HOLD' and not self.owner.get('busy')
                and self.belt_speed == 0. and getattr(self, 'active_goal', None) is None
                and not getattr(self, 'refinement_active', False) and self.now() >= self.clock_last
                and self.feedback_snapshot()['violations'] == ['joint_stamp_ahead_of_clock'])

    def receive_targets(self, message):
        if self.episode is None or message.episode_id!=self.episode:return
        for belief in message.targets:
            if belief.view_id!='overhead' or not belief.camera_only:continue
            b=snapshot(belief)
            if self.bound_track is not None and b['track_id']!=self.bound_track:
                self.failure='Overhead track identity changed during moving inspection'
                return
            if self.overhead is not None and b['capture_ns']<self.overhead['capture_ns']:continue
            advanced=self.overhead is None or b['capture_ns']>self.overhead['capture_ns']
            self.overhead=b
            if advanced:
                self.overhead_frames+=1
                self.overhead_wall=time.monotonic()
                if b['valid']:self.history.append(b)
                if getattr(self,'timing',None) is not None:
                    self.timing.note(b,ns(message.header.stamp),self.now(),context=self.timing_context())

    def target(self):
        self.record_overhead_diagnostic()
        require(self.overhead is not None and self.overhead['episode_id']==self.episode,'No current overhead track')
        require(self.bound_track is None or self.overhead['track_id']==self.bound_track,'Overhead track identity changed')
        require(time.monotonic()-self.overhead_wall<self.mc['maximum_source_wall_age_s'],'Overhead feedback source stopped')
        value=remaining(self.overhead,self.now(),self.mc,self.filter_config)
        self.deadline_diagnostic=dict(value,episode_id=self.episode,track_id=self.overhead['track_id'])
        return value

    def record_overhead_diagnostic(self):
        b=self.overhead
        self.overhead_diagnostic=dict(clock_ns=self.now(),capture_ns=b['capture_ns'] if b else None,
            capture_age_sim_s=(self.now()-b['capture_ns'])/1e9 if b else None,
            source_age_wall_s=time.monotonic()-self.overhead_wall if b else None,
            valid=b['valid'] if b else False,episode_id=b['episode_id'] if b else None,
            track_id=b['track_id'] if b else None,frames_received=self.overhead_frames,
            maximum_capture_age_s=self.mc['maximum_capture_age_s'])

    def prime_overhead(self):
        candidate=None;first_capture=None;last_capture=None;count=0
        self.events.append(dict(state='STARTUP_OVERHEAD',enter_ns=self.now()))
        def ready():
            nonlocal candidate,first_capture,last_capture,count
            require(self.owner.get('owner')=='HOLD' and not self.owner.get('busy'),
                    'Overhead startup requires exclusive HOLD')
            self.record_overhead_diagnostic();b=self.overhead
            if b is None or not b['valid'] or b['episode_id']!=self.episode:
                candidate=first_capture=last_capture=None;count=0;return False
            age=(self.now()-b['capture_ns'])/1e9
            if not 0<=age<self.mc['maximum_capture_age_s']:
                candidate=first_capture=last_capture=None;count=0;return False
            if not 0<=age<=self.mc['overhead_start_maximum_capture_age_s']:return False
            self.target()
            identity=(b['episode_id'],b['track_id'])
            if identity!=candidate:
                candidate=identity;first_capture=last_capture=None;count=0
            if b['capture_ns']!=last_capture:
                if last_capture is not None and (b['capture_ns']-last_capture)/1e9>=self.mc['maximum_tracking_gap_s']:
                    first_capture=None;count=0
                if first_capture is None:first_capture=b['capture_ns']
                last_capture=b['capture_ns'];count+=1
            return count>=self.mc['overhead_start_minimum_captures']
        try:self.until(ready,timeout=15.)
        except TimeoutError as error:
            raise TimeoutError('Overhead startup did not receive three fresh captures: '+json.dumps(self.overhead_diagnostic)) from error
        self.events.append(dict(state='OVERHEAD_READY',enter_ns=self.now(),captures=count,
                                feedback=dict(self.overhead_diagnostic)))

    def until(self, predicate, **kwargs):
        def guarded():
            if getattr(self,'timing',None) is not None and self.timing.status=='PASS':
                self.timing.sample(self.now(),enforce=False,**self.timing_sample_options())
            if kwargs.get('check',True) and self.tracking_active:
                current=self.target()
                if self.operation is not None:
                    elapsed=(self.now()-self.operation['begin_ns'])/1e9
                    budget=self.operation['budget_s']
                    if elapsed>=budget:raise ValueError('Measured inspection movement budget exceeded')
                    needed=budget-elapsed+self.operation['tail_s']
                    if current['remaining_s']<=needed:raise DeadlinePass('deadline_changed_during_inspection')
            return predicate()
        return super().until(guarded,**kwargs)

    def begin_operation(self, name, *, capture=False, returning=False, reserve=True):
        budget=self.costs[name]+(self.mc['capture_budget_s'] if capture else 0.)
        tail=(self.costs['return_assess'] if returning else 0.)+(pick_reserve(self.mc) if reserve else 0.)
        current=self.target()
        self.events.append(dict(state='DEADLINE_CHECK',operation=name,enter_ns=self.now(),
                                budget_s=budget,tail_s=tail,**current))
        if current['remaining_s']<=budget+tail:raise DeadlinePass('insufficient_slack_before_'+name)
        self.operation=dict(name=name,begin_ns=self.now(),budget_s=budget,tail_s=tail)

    def acquire(self, owner, seconds=35., command_type=0):
        if owner!='HOLD' and self.tracking_active:
            seconds=min(seconds,self.target()['remaining_s']-self.mc['margin_s'])
            if seconds<=0:raise DeadlinePass('no_remaining_arm_lease')
        return super().acquire(owner,seconds,command_type)

    def start_transport(self):
        require(getattr(self,'timing',None) is not None and self.timing.status=='PASS','Stationary timing admission has not passed')
        self.prime_overhead()
        self.target();self.bound_track=self.overhead['track_id']
        self.policy_started=(self.now(),time.monotonic())
        self.tracking_active=True;self.belt_speed=float(self.mc['belt_speed_m_s'])
        self.belt.publish(Float64(data=self.belt_speed))
        self.events.append(dict(state='TRANSPORT_STARTED',enter_ns=self.now(),track_id=self.bound_track,
                                commanded_speed_m_s=self.belt_speed))

    def publish_decision(self, result):
        message=Decision();message.header.frame_id='world';message.header.stamp=stamp(self.now())
        message.episode_id=self.episode;message.track_id=self.bound_track
        message.action=result['action'];message.selected_view=result.get('selected_view',result.get('pose',''))
        message.class_id=result.get('class_id','');message.proposed_bin=result.get('proposed_bin','')
        message.confidence=float(result.get('confidence') or 0.);message.class_probabilities=result.get('probabilities',[])
        message.evidence_capture_time=stamp(result.get('evidence_capture_ns',0))
        message.remaining_time_s=float(result.get('remaining_s',0.));message.estimated_service_time_s=float(result.get('estimated_service_time_s',0.))
        message.deadline=stamp(self.now()+round(message.remaining_time_s*1e9))
        message.reason=result['reason'];message.model_sha256=self.model.sha;self.decisions.publish(message)

    def stop(self):
        self.tracking_active=False;self.operation=None
        super().stop()
        if getattr(self,'full_collection_window',None) is not None:
            self.full_collection_window.restore('confirmed_hold')
