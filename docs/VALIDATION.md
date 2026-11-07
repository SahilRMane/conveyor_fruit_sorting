# Validation

## Automated tests

The current suite contains 754 passing tests.
Build and source the workspace first.

```bash
python3 -m unittest discover -s tests
```

The suite covers capture-time geometry, message freshness, filtering, classifier contracts, command handoffs,
physical handling checks, planning constraints, reset state and bounded inspection decisions.
Saved test fixtures include boundary and invalid-input cases.

Run the supervisor subset with:

```bash
python3 -m unittest discover -s tests -p 'test_m4d*.py'
```

## Recorded simulation evaluation

| Evaluation | Result | Conditions |
| --- | --- | --- |
| Static grasp cycle | 20 / 20 | Grip, lift, hold, lower, release and final reset |
| Oracle moving grasp | 20 / 20 | Explicit oracle-only target baseline |
| Camera moving pickup | 19 / 20 | Prepositioned wrist view, 0.03 m/s belt |
| Classified sorting | 20 / 20 | Ten red and ten green nominal sorts, 0.03 m/s belt |
| Covered class handling | Both fixtures passed | No grasp on uncertain class evidence |
| Stationary inspection | Six fixtures passed | Red/green with clear, partial and covered views |
| Moving inspection | Seven cases passed | 0.001 m/s, both classes, clear/partial/covered and late arrival |
| Physical route integration | Both clear-view direct pilots passed | Clear-view class routing, moving grip, placement and reset at 0.001 m/s |

These are recorded simulation results, not physical robot measurements or unrestricted success-rate guarantees.
Classified sorting uses synthetic cues and a fixed initial view.
The integrated physical pilot has a separate scope from the earlier fixed-view sorting batch.

## Preflight

Native checks cover model loading, image geometry, rendering cost, contact behaviour and planning.
Live checks additionally exercise ROS transport, controller execution, command ownership and reset.
Each preflight binds the installed files and generated scenes.
A source or environment change requires a new preflight.

## Current evaluation boundary

Moving active inspection uses a 0.001 m/s engineering configuration.
All seven M4C cases passed functional, timing and final HOLD checks.
M4D.2 connects the inspected route to physical pickup and placement.
Both clear-view direct physical pilots passed.
Full partial-view inspection with physical pickup remains pending.
The ten seeded autonomous inspect/pick/pass episodes remain pending.
Real fruit recognition, hardware transfer and held-out stress benchmarks are future evaluation areas.

See [Running](RUNNING.md) for supported validation commands.

See [Physical inspection sorting](ACTIVE_INSPECTION.md) for the M4D.2 preflight and pilot commands.

M4D retains evidence images in a bounded memory queue during control.
PNG encoding and disk writes occur after control stops.

The red partial-view physical pilot passed inspection, placement and reset.
The matching green partial-view physical pilot remains pending.
