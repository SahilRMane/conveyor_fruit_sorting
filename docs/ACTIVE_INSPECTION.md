# Active inspection and physical sorting

Build the workspace and source the ROS environment. See [Setup](SETUP.md).
Use [Running](RUNNING.md) to generate the moving inspection preflight.

## Physical sorting

Generate passing red and green handoff reports with the inspection preflight.
Use `tools/M4D-handoff/verify_m4c_gate.py` to check the seven inspection cases.
Start a fresh visible inspection launch for each handoff.
Use `tools/M4D-handoff/probe_m4d_handoff.py` with the gate report.
Name the output directories `M4D1-red-handoff-<run>` and `M4D1-green-handoff-<run>`.
Then run the physical preflight.

```bash
ros2 run sorting_supervisor validate_m4d.py \
  --results-root "$PWD/results" --output-dir "$PWD/results/physical-preflight"
```

Set the same preflight path in both terminals.
Use a fresh simulator launch for each pilot.

```bash
ros2 launch sorting_supervisor panda_m4d.launch.py \
  preflight_dir:="$PWD/results/physical-preflight" fixture:=red_partial headless:=false
```

In the second terminal, run the matching pilot.

```bash
ros2 run sorting_supervisor test_m4d.py \
  --preflight-dir "$PWD/results/physical-preflight" \
  --expected-class red_fruit --expected-path inspect \
  --output-dir "$PWD/results/red-physical-pilot"
```

For the green pilot, select the green partial scene and the green class.
Set a new output path for each run.

## Evaluation tools

The scripts under `tools/` use the installed ROS workspace.
Each script lists its arguments with `--help`.
Keep the generated reports and manifests at the paths recorded during the run.

| Tool | Input |
| --- | --- |
| M4 seeded batch | Physical preflight and both physical pilots |
| M5A calibration | Passing M4 batch and native RGB renders |
| M5B moving views | M4 batch and M5A report |
| M5C view utility | M4 batch, M5A report and both M5B probes |
| M5D live selector | M5C report and weights, plus the checked physical preflight |
| M5E learned sorting | M5D suite and left, right and late reports |
| M5F seeded selector | M4 batch, M5D suite and both M5E physical reports |

M5F starts one visible simulator per case.
The camera pipeline supplies the class and target pose.
The evaluator checks grip, bin placement, tray exit and reset.
