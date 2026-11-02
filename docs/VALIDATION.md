# Validation

## Automated tests

The current suite contains 692 passing tests.
Build and source the workspace first.

```bash
python3 -m unittest discover -s tests
```

The suite covers capture-time geometry, message freshness, filtering, classifier contracts, command handoffs,
physical handling checks, planning constraints, reset state and bounded inspection decisions.
Saved test fixtures include boundary and invalid-input cases.

Run the supervisor subset with:

```bash
python3 -m unittest discover -s tests -p 'test_m4c*.py'
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

These are recorded simulation results, not physical robot measurements or unrestricted success-rate guarantees.
Classified sorting uses synthetic cues and a fixed initial view.
Stationary inspection and physical sorting have separate validation scopes.

## Preflight

Native checks cover model loading, image geometry, rendering cost, contact behaviour and planning.
Live checks additionally exercise ROS transport, controller execution, command ownership and reset.
Each preflight binds the installed files and generated scenes.
A source or environment change requires a new preflight.

## Current evaluation boundary

Moving active inspection uses a 0.001 m/s engineering configuration and issues proposals only.
Its broader timing qualification and autonomous connection to physical sorting remain in development.
Real fruit recognition, hardware transfer and held-out stress benchmarks are future evaluation areas.

See [Running](RUNNING.md) for supported validation commands.
