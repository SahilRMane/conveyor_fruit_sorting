"""Evaluate physical contact and transport. No truth enters the controller."""
import json
import time
from types import SimpleNamespace
from std_msgs.msg import String
from m3d_evaluator import SortEvaluator
from test_m4a import Evaluator as WristEvaluator
from ownership_core import require, ns


class IntegratedEvaluator(SortEvaluator):
    def __init__(self, node, executor):
        self.executor = executor
        self.transport = []
        self.wrist_truth = {}
        super().__init__(node)
        self.prepare_trial()
        node.create_subscription(String, '/sorting/ground_truth/conveyor_state', self.receive_transport, 10)

    def receive_transport(self, message):
        value = json.loads(message.data)
        if not self.transport or value['sim_time']-self.transport[-1]['sim_time'] >= .02:
            self.transport.append(value)

    def receive_truth(self, message):
        if message.view_id == 'wrist':
            self.wrist_truth[(message.episode_id, ns(message.header.stamp))] = message
            while len(self.wrist_truth) > 10000:
                self.wrist_truth.pop(next(iter(self.wrist_truth)))
        super().receive_truth(message)

    def until(self, predicate, timeout=30.):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=.01)
            if predicate():
                return
        raise TimeoutError('Integrated evaluator timed out in '+self.phase)

    def check(self, value):
        if self.phase != 'moving_pregrasp' and not (self.phase=='idle' and self.last_placement_phase is None):
            return super().check(value)
        previous=self.phase
        self.phase = 'pregrasp'
        try:
            return super().check(value)
        finally:
            self.phase = previous

    def wrist_checks(self, samples):
        evaluator = SimpleNamespace(truth=self.wrist_truth, failure=None)
        self.until(lambda: WristEvaluator.have_pairs(evaluator, samples), 2.)
        return WristEvaluator.assess(evaluator, samples, .002)

    def transport_checks(self, start_ns, end_ns, speed, closure_ns):
        require(start_ns < closure_ns < end_ns, 'Closure must occur inside the transport interval')
        rows = [r for r in self.transport if start_ns+1_000_000_000 <= round(r['sim_time']*1e9) < end_ns]
        require(len(rows) >= 10, 'Too few continuous conveyor samples')
        free_samples=0
        for row in rows:
            require(row['command_fresh'] and abs(row['requested_speed_m_s']-speed)<1e-9,
                    'Conveyor stopped between inspection and physical lift')
            require(abs(row['force_x_N']) <= 8.000001, 'Conveyor exceeded its force bound')
            # Finger closure constrains tray speed independently of the moving belt.
            if round(row['sim_time']*1e9) < closure_ns and row['support_force_N'] > .5:
                free_samples+=1
                require(abs(row['tray_vx_m_s']-speed)<.0015, 'Supported physical tray speed exceeded its bound')
        require(free_samples >= 10, 'Too few supported free-transport samples')
        return dict(status='PASS', samples=len(rows), free_transport_samples=free_samples,
                    start_ns=start_ns, closure_ns=closure_ns, end_ns=end_ns, belt_speed_m_s=speed)

    def isolation(self, policy_node):
        policies = ('handle_localizer','camera_motion_filter','camera_grasp_scene',policy_node)
        subscriptions = {}
        for name in policies:
            topics = [t for t,_ in self.node.get_subscriber_names_and_types_by_node(name,'/')]
            require(bool(topics), 'Policy node discovery incomplete: '+name)
            require(not any(t.startswith('/sorting/ground_truth/') or t.startswith('/sorting/debug/')
                    or t in ('/sorting/contact_state','/sorting/reset_state') for t in topics),
                    'Evaluator truth entered policy node '+name)
            subscriptions[name] = topics
        require(not any('oracle' in name for name,_ in self.node.get_node_names_and_namespaces()),
                'Oracle node is present in the camera pipeline')
        for topic in ('/sorting/contact_state','/sorting/ground_truth/camera_sample',
                      '/sorting/ground_truth/conveyor_state','/sorting/reset_state'):
            require(all(i.node_name=='m4d_evaluator' for i in self.node.get_subscriptions_info_by_topic(topic)),
                    'Truth subscriber outside the M4D evaluator: '+topic)
        return dict(status='PASS', policy_subscriptions=subscriptions)
