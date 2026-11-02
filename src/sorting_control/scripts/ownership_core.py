"""ROS-independent command ownership, validation and handoff order."""
import math
import threading

NAMES = [f"panda_joint{i}" for i in range(1, 8)]
LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
UPPER = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ns(message):
    return int(message.sec)*1_000_000_000 + int(message.nanosec)


def positions_valid(values):
    return len(values) == 7 and all(math.isfinite(v) and lo <= v <= hi
                                  for v, lo, hi in zip(values, LOWER, UPPER))


def validate_trajectory(trajectory, now_ns, deadline_ns, servo=False):
    require(list(trajectory.joint_names) == NAMES, "Unexpected arm joint order")
    require(bool(trajectory.points), "Empty trajectory")
    require(deadline_ns > now_ns, "Command deadline expired")
    previous = -1
    for point in trajectory.points:
        require(positions_valid(point.positions), "Invalid joint positions")
        for field in (point.velocities, point.accelerations, point.effort):
            require(len(field) in (0, 7) and all(math.isfinite(v) for v in field), "Invalid trajectory fields")
        stamp = ns(point.time_from_start)
        require(stamp >= 0 and stamp > previous, "Trajectory times must increase")
        previous = stamp
    if not servo:
        require(ns(trajectory.header.stamp) == 0, "Use an immediate trajectory with a zero header stamp")
        require(previous > 0 and now_ns+previous+500_000_000 < deadline_ns, "Trajectory cannot finish before its deadline")


class StoppedFeedbackWindow:
    """Require new stopped samples across time; a polling loop is not a sample."""
    def __init__(self, hold_start_ns, initial_sequence, velocity_limit,
                 stable_span_ns=200_000_000, minimum_samples=5, minimum_hold_ns=450_000_000):
        self.hold_start_ns = hold_start_ns
        self.last_sequence = initial_sequence
        self.last_stamp_ns = None
        self.velocity_limit = velocity_limit
        self.stable_span_ns = stable_span_ns
        self.minimum_samples = minimum_samples
        self.minimum_hold_ns = minimum_hold_ns
        self.first_stopped_stamp_ns = None
        self.samples = 0

    def clear(self):
        self.first_stopped_stamp_ns = None
        self.samples = 0

    def observe(self, sequence, stamp_ns, now_ns, velocities, fresh):
        if not fresh:
            self.clear()
            self.last_sequence = max(self.last_sequence, sequence)
            self.last_stamp_ns = stamp_ns
            return False
        if sequence > self.last_sequence:
            self.last_sequence = sequence
            if self.last_stamp_ns is not None and stamp_ns < self.last_stamp_ns:
                self.clear()
            new_stamp = self.last_stamp_ns is None or stamp_ns > self.last_stamp_ns
            self.last_stamp_ns = stamp_ns
            stopped = (len(velocities) == 7 and all(math.isfinite(v) and abs(v) < self.velocity_limit for v in velocities)
                       and stamp_ns >= self.hold_start_ns)
            if not stopped:
                self.clear()
            elif new_stamp:
                if self.first_stopped_stamp_ns is None:
                    self.first_stopped_stamp_ns = stamp_ns
                self.samples += 1
        return (self.first_stopped_stamp_ns is not None and self.samples >= self.minimum_samples
                and self.last_stamp_ns-self.first_stopped_stamp_ns >= self.stable_span_ns
                and now_ns-self.hold_start_ns >= self.minimum_hold_ns)

    def snapshot(self):
        span = 0 if self.first_stopped_stamp_ns is None or self.last_stamp_ns is None else self.last_stamp_ns-self.first_stopped_stamp_ns
        return {"stopped_samples": self.samples, "required_samples": self.minimum_samples,
                "stopped_sample_span_s": max(0, span)/1e9, "required_span_s": self.stable_span_ns/1e9}


class Ownership:
    def __init__(self, adapter, maximum_lease_s=60.0):
        self.adapter = adapter
        self.lock = threading.RLock()
        self.owner = "HOLD"
        self.epoch = 0
        self.deadline_ns = 0
        self.command_type = 0
        self.busy = False
        self.ready = False
        self.maximum_lease_ns = round(maximum_lease_s*1e9)
        self.activation_ns = 0
        self.reason = "startup"
        self.stats = {"planned_accepted": 0, "planned_rejected": 0,
                      "servo_input_accepted": 0, "servo_input_rejected": 0,
                      "servo_output_forwarded": 0, "servo_output_dropped": 0,
                      "overlap_violations": 0}

    def event(self, event, **details):
        self.adapter.event(event, owner=self.owner, epoch=self.epoch, **details)

    def allowed(self, owner, epoch, deadline_ns, now_ns):
        return (self.ready and not self.busy and self.owner == owner and self.epoch == epoch
                and now_ns < deadline_ns <= self.deadline_ns)

    def snapshot(self):
        with self.lock:
            return {"owner": self.owner, "epoch": self.epoch, "deadline_ns": self.deadline_ns,
                    "busy": self.busy, "ready": self.ready, "reason": self.reason,
                    "command_type": self.command_type, "stats": self.stats.copy()}

    def reserve_stop(self, reason):
        with self.lock:
            if self.busy:
                return None
            self.epoch += 1
            self.owner = "SWITCHING"
            self.busy = True
            self.deadline_ns = 0
            self.reason = reason
            self.event("owner_transition_begin", target="HOLD", reason=reason)
            return self.epoch

    def invalidate_pending_transition(self, reason):
        with self.lock:
            if not self.busy:
                return self.reserve_stop(reason)
            # A reset can occur while a service is awaited. The pending task
            # must never activate its destination with a previous-episode seed.
            self.epoch += 1
            self.reason = reason
            self.event("owner_transition_invalidated", reason=reason)
            return None

    async def change(self, owner, deadline_ns=0, command_type=0, reason="requested"):
        with self.lock:
            require(owner in {"HOLD", "PLANNED", "SERVO"}, "Unknown owner")
            require(not self.busy, "An owner transition is already running")
            now = self.adapter.now_ns()
            if owner != "HOLD":
                require(now < deadline_ns <= now+self.maximum_lease_ns, "Invalid owner deadline")
            if owner == "SERVO":
                require(command_type in (0, 1), "Use JOINT_JOG or TWIST")
            self.epoch += 1
            ticket = self.epoch
            self.owner = "SWITCHING"
            self.busy = True
            self.deadline_ns = 0
            self.reason = reason
            self.event("owner_transition_begin", target=owner, reason=reason)
        return await self.finish(ticket, owner, deadline_ns, command_type)

    async def finish(self, ticket, target="HOLD", deadline_ns=0, command_type=0):
        try:
            if not self.ready:
                await self.adapter.wait_ready()
            await self.adapter.pause_servo(True)
            self.event("servo_paused")
            await self.adapter.cancel_plan()
            self.event("planned_cancel_confirmed")
            if self.ready:
                # Stop existing sources before waiting for delayed feedback.
                await self.adapter.wait_ready()
            seed = await self.adapter.hold_and_wait()
            self.event("arm_stopped", measured_positions=seed)
            with self.lock:
                require(self.epoch == ticket, "Owner transition was invalidated")
            if target == "SERVO":
                await self.adapter.set_servo_type(command_type)
                await self.adapter.pause_servo(False)
                self.event("servo_reinitialized_from_feedback")
            with self.lock:
                require(self.epoch == ticket, "Owner transition was invalidated")
                if target != "HOLD":
                    require(self.adapter.now_ns() < deadline_ns, "Owner deadline expired during handoff")
                self.owner = target
                self.deadline_ns = deadline_ns if target != "HOLD" else 0
                self.command_type = command_type
                self.activation_ns = self.adapter.now_ns()
                self.busy = False
                self.ready = True
                self.event("owner_activated", measured_positions=seed)
            return True
        except Exception as error:
            with self.lock:
                self.owner = "FAULT"
                self.deadline_ns = 0
                self.busy = False
                self.ready = False
                self.epoch += 1
                self.reason = str(error)
                self.event("owner_transition_failed", error=str(error))
            await self.adapter.failure_stop()
            return False
