"""Small deterministic software example, not empirical or recovery evidence.

No optimizer is called, no participants are read, and no inference is performed.
The example exercises the actual scoring and parameter-definition functions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import sys
import numpy as np
import pandas as pd
import scipy
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit.t5_memory import model as t5_model
from fit.t6 import pipeline as t6_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    xy = np.array([[200, 200], [250, 210], [300, 250], [240, 230], [220, 200], [280, 260]], dtype=float)
    trials = [dict(trial_number=i+1, emotion=e, x=xy[:, 0]+i*10,
                   y=xy[:, 1], t=np.arange(6, dtype=float)*.1, n=6, n_original=6)
              for i, e in enumerate(['positive', 'negative', 'neutral', 'complex'])]
    data5 = dict(width=960.0, height=720.0, trials=trials)
    theta = np.array([.5, 120., 80.] * 4)
    score5 = float(t5_model.loglik_picture_only(data5, theta))
    metrics5 = t5_model.compute_interpretable_metrics(data5, theta)
    assert math.isfinite(score5)
    for values in metrics5.values():
        assert math.isclose(values['T_half'], .5 * math.log(2), abs_tol=1e-14)
        assert 0 <= values['H_bar_norm'] <= 1
        assert math.isclose(values['I_focus'], 1-math.exp(-50**2/(2*80**2)), abs_tol=1e-14)
    data6 = dict(gaze_x=np.array([960., 300., 960., 1600.]),
                 gaze_y=np.array([540., 500., 500., 500.]),
                 positive=np.array([0., 1., 0., 0.]),
                 nagetive=np.array([0., 0., 1., 0.]),
                 neutral=np.array([0., 0., 0., 1.]),
                 t_from_trial_start=np.array([0., 200., 600., 1600.]),
                 t_from_picture_start=np.array([-100., 100., 500., 1500.]))
    score6 = t6_model.compute_log_likelihood(data6, t6_model.DEFAULT_INIT_PARAMS)
    angle = float(t6_model.px_to_visual_angle_deg(120))
    assert math.isfinite(score6) and 0 < angle < 180
    code_fingerprints = {
        str(Path(module.__file__).relative_to(ROOT)): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (t5_model, t6_model)
    }
    report = dict(status='passed', scope='deterministic scoring and formula software check only',
        raw_participants_read=0, optimization_runs=0, new_statistical_tests=0,
        T5_synthetic_observations=24, T6_synthetic_observations=4,
        T5_score=score5, T6_empirical_score=float(score6), T5_example_metrics=metrics5,
        central_120px_degrees=angle, model_code_sha256=code_fingerprints,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        historical_execution_environment='not established by this check')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'raw_participants_read', 'optimization_runs', 'new_statistical_tests', 'environment']}))


if __name__ == '__main__':
    main()
