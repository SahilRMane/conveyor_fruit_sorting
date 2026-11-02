"""Save trial evidence and apply the 18/20 acceptance rule."""
import json
from pathlib import Path

NOMINAL_CYCLES = 20
REQUIRED_SUCCESSES = 18


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def assess(requested, trials, interrupted=False, infrastructure_error=None):
    completed = [t for t in trials if t.get("status") in {"PASS", "FAIL"}]
    passes = sum(t["status"] == "PASS" for t in completed)
    failures = len(completed) - passes
    resets_ok = all(t.get("reset_status") == "PASS" for t in trials)
    full = len(completed) == requested and len(trials) == requested
    acceptance = requested == NOMINAL_CYCLES and full and resets_ok and passes >= REQUIRED_SUCCESSES
    if interrupted or infrastructure_error:
        status = "ABORTED"
    elif not full:
        status = "INCOMPLETE"
    elif requested != NOMINAL_CYCLES:
        status = "SMOKE_PASS" if failures == 0 and resets_ok else "FAIL"
    else:
        status = "PASS" if acceptance else "FAIL"
    return {"status": status, "requested_cycles": requested, "completed_cycles": len(completed),
            "successful_cycles": passes, "failed_cycles": failures,
            "not_completed_cycles": max(0, requested-len(completed)),
            "reset_verified": bool(trials) and resets_ok and not infrastructure_error,
            "acceptance_target": {"cycles": NOMINAL_CYCLES, "minimum_successes": REQUIRED_SUCCESSES},
            "acceptance_passed": acceptance and status == "PASS",
            "infrastructure_error": infrastructure_error, "interrupted": interrupted}
