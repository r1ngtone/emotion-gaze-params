"""Write invented eye/record CSV pairs for software checks."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "timestamp", "bino_eye_gaze_position_x", "bino_eye_gaze_position_y",
    "bino_eye_valid", "trigger",
]
EMOTIONS = ("positive", "negative", "neutral")
REGIONS = ((120, 350), (710, 350), (1300, 350))


def example_inputs(task: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    eye, records = [], []
    for trial in range(1, 25 if task == "T5" else 41):
        start = trial * 3_000_000_000
        code = trial * 100
        fixation = rng.normal([960, 540], [27, 23], size=(8, 2))
        for index, (x, y) in enumerate(fixation):
            eye.append((start + index * 25_000_000, x, y, 1, code if index == 0 else 0))

        if task == "T5":
            emotion = (EMOTIONS + ("complex",))[(trial - 1) % 4]
            records.append({"试次轮次": trial, "刺激图片情绪类型": emotion})
            start_xy = rng.uniform([650, 330], [1250, 740])
            switches = rng.choice([-1, 0, 1], size=(200, 2))
            locations = np.clip(start_xy + switches.cumsum(axis=0) * 13 + rng.normal(0, 22, size=(200, 2)),
                                [482, 182], [1438, 898])
            step = 8_000_000
        else:
            layout = np.roll(EMOTIONS, trial % 3)
            row = {}
            for side, origin, emotion in zip("左中右", REGIONS, layout):
                row[f"{side}图片坐标"] = str(origin)
                row[f"{side}图片大小"] = "(500, 380)"
                row[f"{side}情绪类型"] = emotion
            records.append(row)
            locations = []
            for index in range(56):
                early = index / 55
                region = rng.choice(3, p=[0.46 - 0.20 * early, 0.28 + 0.10 * early, 0.26 + 0.10 * early])
                origin_x, origin_y = REGIONS[region]
                locations.append(rng.normal([origin_x + 250, origin_y + 190], [47, 42]))
            step = 30_000_000

        picture = start + 240_000_000
        for index, (x, y) in enumerate(locations):
            eye.append((picture + index * step, x, y, int(index % 29 != 11), code + 1 if index == 0 else 0))
        eye.append((picture + len(locations) * step, 960, 540, 1, code + 2))
    if task == "T6":
        eye.append((eye[-1][0] + 30_000_000, 960, 540, 1, 4003))
    return pd.DataFrame(eye, columns=COLUMNS), pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    manifest = []
    for seed in (51, 72):
        subject_id = f"synthetic-{seed}"
        for task in ("T5", "T6"):
            eye, records = example_inputs(task, seed)
            eye_name = f"{subject_id}-{task}-eye.csv"
            records_name = f"{subject_id}-{task}-records.csv"
            eye.to_csv(args.output_dir / eye_name, index=False)
            records.to_csv(args.output_dir / records_name, index=False)
            manifest.append(dict(subject_id=subject_id, task_type=task, eyetrack_path=eye_name,
                                 records_path=records_name, status="success"))
    pd.DataFrame(manifest).to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"Synthetic inputs ready: {args.output_dir}")


if __name__ == "__main__":
    main()
