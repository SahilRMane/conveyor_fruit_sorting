# Running the simulation

Build the workspace with [Setup](SETUP.md).
Use the terminal environment from that guide in every terminal.
Start only one simulator launch at a time.

## Panda viewer

```bash
ros2 launch sorting_description panda_mujoco.launch.py headless:=false
```

This opens the Panda scene and starts the joint-state and trajectory controllers.
Stop the launch before starting another workflow.

## Classified moving-tray sorting

The sorting workflow uses synthetic red/green class fixtures and physical tray contact.
Prepare native checks while the simulator is stopped.

```bash
RUN_ROOT="$PWD/results/run-$(date +%Y%m%d-%H%M%S-%N)"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > results/current-run.txt

OPENBLAS_NUM_THREADS=1 \
  ros2 run sorting_perception validate_m3d_files.py \
  --components-only --output-dir "$RUN_ROOT/components"

ros2 run sorting_perception probe_m3d_planning.py \
  --preflight-dir "$RUN_ROOT/components" \
  --output-dir "$RUN_ROOT/planning"

OPENBLAS_NUM_THREADS=1 \
  ros2 run sorting_perception validate_m3d_files.py \
  --output-dir "$RUN_ROOT/preflight"
```

Continue after all three checks return PASS.

Terminal 1:

```bash
ros2 launch sorting_control panda_m3d.launch.py \
  scene_fixture:=red_clear headless:=false
```

Terminal 2, with the workspace environment sourced:

```bash
RUN_ROOT="$(cat results/current-run.txt)"

ros2 run sorting_perception test_m3d.py \
  --preflight-dir "$RUN_ROOT/preflight" \
  --planning-probe-dir "$RUN_ROOT/planning" \
  --expected-class red_fruit --trials 1 --summary-only \
  --output-dir "$RUN_ROOT/red-sort"
```

The test checks classification, physical pickup, transfer, placement and final reset.
For the other class, use `scene_fixture:=green_clear`, `--expected-class green_fruit`
and a new output directory.
Nominal batches accept `--trials 10`.
Covered fixtures are `red_covered` and `green_covered`.
Their tests require `--expect-pass` because the intended action is to let the tray pass.

## Stationary active inspection

Generate a new inspection preflight while the simulator is stopped:

```bash
RUN_ROOT="$PWD/results/inspection-$(date +%Y%m%d-%H%M%S-%N)"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > results/current-inspection.txt

OPENBLAS_NUM_THREADS=1 \
  ros2 run sorting_supervisor validate_m4b.py \
  --output-dir "$RUN_ROOT/preflight"
```

Terminal 1:

```bash
RUN_ROOT="$(cat results/current-inspection.txt)"

ros2 launch sorting_supervisor panda_m4b.launch.py \
  preflight_dir:="$RUN_ROOT/preflight" \
  fixture:=red_partial headless:=false
```

Terminal 2:

```bash
RUN_ROOT="$(cat results/current-inspection.txt)"

ros2 run sorting_supervisor test_m4b.py \
  --preflight-dir "$RUN_ROOT/preflight" \
  --expected-class red_fruit --expected-action PICK_READY \
  --expected-inspections 1 \
  --output-dir "$RUN_ROOT/M4B-red_partial"
```

Restart Terminal 1 for each fixture.
Set the fixture, expected class, expected action, inspection count and output name from this table.

| Fixture | Expected class | Expected action | Inspections |
| --- | --- | --- | --- |
| red_clear | red_fruit | PICK_READY | 0 |
| green_clear | green_fruit | PICK_READY | 0 |
| red_partial | red_fruit | PICK_READY | 1 |
| green_partial | green_fruit | PICK_READY | 1 |
| red_covered | red_fruit | PASS | 1 |
| green_covered | green_fruit | PASS | 1 |

Use output directories named `M4B-<fixture>` under the same run root.
These six runs supply measured motion costs for moving inspection.
A PICK_READY proposal in this workflow does not command a grasp.

## Moving inspection

This engineering workflow uses continuous overhead tracking while the wrist moves.
The belt speed is 0.001 m/s.
Complete the six stationary fixture tests above with the same installed model first.

Stop the simulator and generate the moving preflight:

```bash
RUN_ROOT="$(cat results/current-inspection.txt)"

OPENBLAS_NUM_THREADS=1 \
  ros2 run sorting_supervisor validate_m4c.py \
  --m4b-results-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/moving-preflight"
```

Terminal 1:

```bash
RUN_ROOT="$(cat results/current-inspection.txt)"

ros2 launch sorting_supervisor panda_m4c.launch.py \
  preflight_dir:="$RUN_ROOT/moving-preflight" \
  fixture:=red_partial arrival:=upstream headless:=false
```

Terminal 2:

```bash
RUN_ROOT="$(cat results/current-inspection.txt)"

ros2 run sorting_supervisor test_m4c.py \
  --preflight-dir "$RUN_ROOT/moving-preflight" \
  --expected-class red_fruit --expected-path inspect \
  --output-dir "$RUN_ROOT/moving-red-partial"
```

A full upstream inspection includes conveyor entry, observation, wrist movement and return.
Use a fresh simulator launch and a new output directory for every run.

| Fixture | Arrival | Expected class | Expected path |
| --- | --- | --- | --- |
| red_partial | upstream | red_fruit | inspect |
| green_partial | upstream | green_fruit | inspect |
| red_clear | upstream | red_fruit | direct |
| green_clear | upstream | green_fruit | direct |
| red_covered | upstream | red_fruit | unresolved |
| green_covered | upstream | green_fruit | unresolved |
| red_clear | late | red_fruit | deadline |

This workflow validates inspection proposals.
See [Active inspection](ACTIVE_INSPECTION.md) for the physical sorting workflow.

## Outputs

Commands write reports, camera frames and timing records under `results/`.
Each output directory must be new.
Preflight data binds source versions, model files and generated scenes.
Rebuild and create a new preflight after a source or environment change.
