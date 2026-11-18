"""Explicit-input entry point for the unchanged formal model functions.

This entry point does not search participant folders, run a classifier, impute
samples, or change optimizers. Raw inputs must be approved, correctly decoded
formal exports matching the contracts in README.md.
"""
from pathlib import Path
import argparse
import json
import pandas as pd
from fit.t5_memory import model as t5_model, pipeline as t5_preprocessing
from fit.t6 import pipeline as t6_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['T5', 'T6'])
    parser.add_argument('--eyetrack', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output filename; existing results are preserved.')
    eye = pd.read_csv(args.eyetrack, encoding='utf-8-sig')
    records = pd.read_csv(args.records, encoding='utf-8-sig')
    if args.task == 'T5':
        stages = t5_preprocessing.run_memory_pipeline_from_dataframes(eye, records)
        result = t5_model.run_model_for_a5_df(stages['A5'], timestamp_unit_scale=1e9,
                                            resample_ratio=.07, M=12, N=9, r_focus=50.0)
        output = dict(task='T5', sampling_ratio=.07, result=result['flat_result'])
    else:
        base, detected_trials = t6_model.build_base_frame(eye)
        frame = t6_model.annotate_spatial_and_emotion(base, records)
        _, data = t6_model.prepare_model_data(frame, timestamp_unit_scale=1e6)
        fit = t6_model.optimize_free_parameters(data)
        fitted = fit['best_params']
        reported = {f'Tmax_{e}': fitted[f'Tmax{i}'] for i, e in enumerate(['positive', 'negative', 'neutral'], 1)}
        reported.update(pm_positive=fit['Pm1_positive'], pm_negative=fit['Pm2_negative'],
                        pm_neutral=fit['Pm3_neutral'], dc=float(t6_model.px_to_visual_angle_deg(fitted['dc'])))
        output = dict(task='T6', reported=reported, internal_parameters=fitted,
                      empirical_objective=fit['best_log_likelihood'], optimizer_success=fit['success'],
                      optimizer_message=fit['message'], detected_trials=int(detected_trials))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(payload + '\n')
    print('Wrote explicit-input model result; no diagnostic classification was run.')


if __name__ == '__main__':
    main()
