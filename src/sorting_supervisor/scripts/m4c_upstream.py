"""Join diagnostic records by exact capture identity. Never change target data."""
from collections import OrderedDict,Counter
from copy import deepcopy


def camera_delivery_timing(local):
    """Read Fast DDS system timestamps without mixing them with source ROS time."""
    raw=local.get('camera_delivery',{})
    if not isinstance(raw,dict):return dict(status='MALFORMED_METADATA',raw=deepcopy(raw))
    names=('source_timestamp_system_ns','received_timestamp_system_ns','callback_entry_system_ns')
    present={k:v for k,v in raw.items() if k in names and type(v) is int and v>0}
    missing=[k for k in names if k not in present]
    result=dict(status='UNAVAILABLE' if len(missing)==len(names) else 'PARTIAL' if missing else 'COMPLETE',
        missing=missing,raw=deepcopy(raw),clock_domain='System time / Unix epoch; separate from ROS and monotonic time',
        scope='Publication is the middleware source timestamp, not the camera publish() entry. '
              'Fast DDS receipt is when the sample enters reader history. Receipt-to-callback includes '
              'queueing, executor scheduling, take/deserialization and callback dispatch. '
              'Positive system-clock steps during an older queued sample may remain undetected.')
    if raw.get('rmw_implementation') not in ('rmw_fastrtps_cpp','rmw_fastrtps_dynamic_cpp'):
        if raw:result['status']='UNVALIDATED_RMW'
        return result
    before=raw.get('callback_entry_monotonic_before_ns');after=raw.get('callback_entry_monotonic_after_ns')
    if type(before) is not int or type(after) is not int or after<before:
        result['status']='INVALID_CLOCK_SAMPLE';return result
    result['clock_pair_span_wall_s']=(after-before)/1e9
    # One millisecond is a diagnostic clock-sampling limit, not an active motion limit.
    if after-before>1_000_000:
        result['status']='CLOCK_SAMPLE_SPAN_EXCEEDED';return result
    delta=raw.get('system_monotonic_delta_ns');uncertainty=raw.get('offset_comparison_uncertainty_ns')
    if type(delta) is int and type(uncertainty) is int and abs(delta)>1_000_000+uncertainty:
        result['status']='SYSTEM_CLOCK_DISCONTINUITY';return result
    pairs={
        'publication_to_middleware_receipt_system_s':('source_timestamp_system_ns','received_timestamp_system_ns'),
        'middleware_receipt_to_callback_system_s':('received_timestamp_system_ns','callback_entry_system_ns'),
        'publication_to_callback_system_s':('source_timestamp_system_ns','callback_entry_system_ns')}
    result['intervals']={k:(present[end]-present[begin])/1e9 for k,(begin,end) in pairs.items()
                         if begin in present and end in present}
    result['negative_system_intervals']=[k for k,v in result['intervals'].items() if v<0]
    if result['negative_system_intervals']:result['status']='CLOCK_OR_ORDER_MISMATCH'
    return result


class UpstreamTrace:
    def __init__(self):
        self.rows=OrderedDict();self.pinned=set();self.counts=Counter()

    def pin(self,samples):
        self.pinned={(s['episode_id'],'overhead',s['capture_ns']) for s in samples if s is not None}
        self.trim()

    def trim(self):
        # shortcut: one local test process, at most 128 recent and four pinned captures.
        while len(self.rows)>128+len(self.pinned):
            old=next(k for k in self.rows if k not in self.pinned)
            del self.rows[old]

    def note(self,row,stage,episode):
        if (not isinstance(row,dict) or row.get('version')!=1 or row.get('stage')!=stage
                or row.get('episode_id')!=episode or row.get('view_id')!='overhead'
                or any(type(row.get(k)) is not int or row[k]<0 for k in ('capture_ns','episode_id','observation_sequence'))):
            self.counts['rejected']+=1;return
        key=(row['episode_id'],row['view_id'],row['capture_ns'])
        entry=self.rows.setdefault(key,{})
        if stage in entry:self.counts['duplicate_'+stage]+=1;return
        entry[stage]=deepcopy(row);self.counts[stage]+=1;self.trim()

    def capture(self,sample):
        key=(sample['episode_id'],'overhead',sample['capture_ns'])
        rows=self.rows.get(key,{})
        missing=[s for s in ('localizer','filter') if s not in rows]
        result=dict(identity=dict(episode_id=key[0],view_id=key[1],capture_ns=key[2]),
                    status='MISSING' if missing else 'COMPLETE',missing=missing,stages=deepcopy(rows))
        if missing:return result
        local,filtered=rows['localizer'],rows['filter']
        if (local['observation_sequence']!=filtered['observation_sequence']
                or filtered.get('track_id')!=sample['track_id']):
            result['status']='IDENTITY_MISMATCH';return result
        result['camera_delivery']=camera_delivery_timing(local)
        points=dict(local,**filtered,executor_receive_wall_ns=round(sample['receipt_wall_s']*1e9))
        pairs={
            'image_to_first_attempt_wall_s':('image_arrival_wall_ns','first_attempt_wall_ns'),
            'first_attempt_to_metadata_ready_wall_s':('first_attempt_wall_ns','metadata_ready_wall_ns'),
            'metadata_ready_to_tf_ready_wall_s':('metadata_ready_wall_ns','tf_ready_wall_ns'),
            'image_preparation_wall_s':('image_prepare_start_wall_ns','image_prepare_end_wall_ns'),
            'successful_tf_lookup_wall_s':('tf_lookup_start_wall_ns','tf_ready_wall_ns'),
            'tf_ready_to_localization_end_wall_s':('tf_ready_wall_ns','localization_end_wall_ns'),
            'localization_compute_wall_s':('localization_start_wall_ns','localization_end_wall_ns'),
            'localization_end_to_observation_publish_wall_s':('localization_end_wall_ns','observation_publish_enter_wall_ns'),
            'observation_publish_to_filter_receive_wall_s':('observation_publish_enter_wall_ns','filter_receive_wall_ns'),
            'filter_receive_to_select_wall_s':('filter_receive_wall_ns','filter_select_wall_ns'),
            'filter_update_wall_s':('filter_select_wall_ns','filter_update_end_wall_ns'),
            'filter_update_end_to_publish_wall_s':('filter_update_end_wall_ns','filter_publish_enter_wall_ns'),
            'filter_publish_to_executor_receive_wall_s':('filter_publish_enter_wall_ns','executor_receive_wall_ns'),
            'image_to_executor_receive_wall_s':('image_arrival_wall_ns','executor_receive_wall_ns'),
        }
        absent=sorted({p for pair in pairs.values() for p in pair if type(points.get(p)) is not int})
        if absent:
            result.update(status='MISSING_TIMESTAMPS',missing=absent);return result
        result['intervals']={name:(points[end]-points[begin])/1e9 for name,(begin,end) in pairs.items()}
        if type(local.get('image_arrival_sim_ns')) is int:
            result['image_arrival_capture_age_sim_s']=(local['image_arrival_sim_ns']-key[2])/1e9
        result['negative_wall_intervals']=[k for k,v in result['intervals'].items() if v<0]
        if result['negative_wall_intervals']:result['status']='CLOCK_OR_ORDER_MISMATCH'
        return result

    def decorate(self,sample):
        if sample is None:return None
        return dict(sample,upstream_trace=self.capture(sample))

    def report(self):
        return dict(counts=dict(self.counts),cached_captures=len(self.rows),pinned_captures=len(self.pinned),
            maximum_cached_captures=128+len(self.pinned),
            scope='Diagnostic only. Wall intervals use one host monotonic clock. Local ROS clock ages are separate. '
                  'Metadata and TF readiness mean first successful process attempt, not their earliest availability. '
                  'Time before image arrival combines rendering, transport, callback dispatch and ROS clock delivery. '
                  'Diagnostic messages add work; missing traces never authorize or reject motion.')
