"""One-extra-view confidence state machine. No fixture labels or robot commands."""
import math


class SingleViewPolicy:
    def __init__(self, episode, begin_ns, config):
        self.episode, self.barrier, self.config = episode, begin_ns, config
        self.state, self.inspections, self.last_capture = 'ASSESS', 0, 0
        self.history = []

    def start_inspection(self, now_ns):
        if self.state != 'INSPECT' or self.inspections:
            raise ValueError('Only one requested inspection is permitted')
        self.inspections = 1; self.barrier = now_ns

    def observe(self, *, episode, pose, capture_ns, now_ns, probabilities):
        if self.state not in ('ASSESS', 'INSPECT'):
            raise ValueError('Decision is terminal')
        if episode != self.episode:
            raise ValueError('Episode changed')
        required_pose = 'assess' if self.state == 'ASSESS' else self.config['inspection_view']
        if pose != required_pose or (self.state == 'INSPECT' and self.inspections != 1):
            raise ValueError('Observation does not belong to the current view/state')
        p = list(probabilities)
        valid = (len(p) == len(self.config['class_ids']) and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                 and abs(sum(p)-1.) < 1e-6 and capture_ns > max(self.barrier, self.last_capture)
                 and 0 <= now_ns-capture_ns <= self.config['maximum_capture_age_s']*1e9)
        before = self.state
        if not valid:
            decision = dict(action='PASS', reason='invalid_or_stale_class_evidence')
        else:
            self.last_capture = capture_ns; index = max(range(len(p)), key=p.__getitem__)
            if p[index] >= self.config['acceptance_confidence']:
                label = self.config['class_ids'][index]
                decision = dict(action='PICK_READY', class_id=label, proposed_bin=self.config['bins'][label],
                                confidence=p[index], probabilities=p, reason='fresh_single_capture_class_evidence')
            elif self.inspections == 0:
                decision = dict(action='INSPECT', selected_view=self.config['inspection_view'], reason='uncertain_initial_view')
            else:
                decision = dict(action='PASS', reason='uncertain_after_one_extra_view')
        self.state = decision['action']
        decision.update(evidence_capture_ns=capture_ns, episode_id=episode, pose=pose,
                        confidence=max(p) if valid else None, probabilities=p if valid else [],
                        proposal_only=True, physical_pickup_permitted=False)
        self.history.append(dict(from_state=before, to_state=self.state, **decision))
        return decision
