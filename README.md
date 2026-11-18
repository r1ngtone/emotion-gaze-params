# Emotion-gaze trajectory parameterization

Reproduction code for the revised V7.1 manuscript. T5 estimates gaze-history and spatial parameters from a single-picture task. T6 estimates timing, allocation, and central spread from a three-picture task. [Manuscript result index](REPRODUCIBILITY.md) maps the analyses to their evidence and inputs.

## Set up

The software checks were validated with Python 3.11.14.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python example/minimal_scoring_example.py --output out/scoring.json
python -m unittest discover -s fit/t5_memory -v
```

The scoring example checks the actual model functions and report definitions on invented observations. It does not optimize or read study data.

## Fit one session

Generate two entirely synthetic participant pairs and run both formal fitting entry points:

```sh
python example/generate_synthetic_inputs.py --output-dir out/synthetic
python fit_subject.py T5 --eyetrack out/synthetic/synthetic-51-T5-eye.csv --records out/synthetic/synthetic-51-T5-records.csv --output out/t5.json
python fit_subject.py T6 --eyetrack out/synthetic/synthetic-51-T6-eye.csv --records out/synthetic/synthetic-51-T6-records.csv --output out/t6.json
```

Replace the two CSV paths with approved formal exports to fit a study session. Output JSON includes optimizer status and the fitted values. A new output filename is required.

Both eye tables require `timestamp`, `bino_eye_gaze_position_x`, `bino_eye_gaze_position_y`, `bino_eye_valid`, and `trigger`. Input gaze coordinates are in screen pixels. A binocular validity flag of 1 marks valid observations. The formal analysis expects fixation, image onset, and blank events at `100 × trial + {0,1,2}`.

- **T5.** Trial records require `试次轮次` (trial number) and `刺激图片情绪类型` (positive, negative, neutral or complex). The picture is the 960 × 720 rectangle at (480, 180). Image-stage time is measured from the first image-stage sample, before device-validity and picture-area filters, and raw timestamp differences are divided by `1e9` to obtain seconds. The formal entry point deterministically retains 7% per trial, with at least one point in each nonempty trial. It uses a 12 × 9 grid, the production initial value `(0.5, 120, 80)` per emotion, and bounded L-BFGS-B. It outputs the fitted values and `T_half`, `H_bar_norm`, and `I_focus`.
- **T6.** The trial record rows are in order. Each of the three sides `左`, `中`, `右` requires columns with suffixes `图片坐标` (top-left pair), `图片大小` (size pair), and `情绪类型` (positive, negative or neutral). The formal export includes end marker `4003`. The picture time origin is the first valid, on-screen image-stage sample. Timestamp differences are divided by `1e6` for milliseconds. The formal fit uses the empirical mixed score and reports `Tmax`, `pm`, and `dc` (degrees of visual angle).

The original [T5 A5 demonstration](example/generate_synthetic_demo.py) illustrates the model at full sampling density on internally generated data.

## Recovery and Figure 3

```sh
python recovery/run_t5_parameter_recovery.py --smoke --output-dir out/t5-recovery-smoke
python recovery/run_t6_r4c.py --truth-ids T6OW-01 T6OW-03 --repeats 1 --n-starts 2 --output-dir out/t6-recovery-smoke
python analysis/run_timeblock_permutation_n118.py --manifest out/synthetic/manifest.csv --output-dir out/timeblock-smoke --permutations 3 --workers 1
```

These commands are small software checks. See the [result index](REPRODUCIBILITY.md) for the original settings and expected study summaries.

The time-block entry takes a CSV with one successful row per task and opaque participant ID. Required columns are `subject_id`, `task_type`, `eyetrack_path`, `records_path`, and `status`. Paths resolve relative to the manifest unless absolute. It defaults to four folds, 250 ms blocks, 1,000 permutations and seed 20260727. Outputs and checkpoints are stored under the chosen output directory. Run with the same inputs and settings to resume, or choose a fresh directory when inputs or settings change.

## Materials

The task directory contains archived task scripts and trial lists. Original stimulus images and hardware dependencies are separate. [Availability statements](STATEMENTS.md) describe the scope of this package. The [v1.0 tag](https://github.com/r1ngtone/emotion-gaze-params/tree/v1.0) preserves the previously released exploratory scripts and the historical T6 `16 × 100` recovery experiment.
