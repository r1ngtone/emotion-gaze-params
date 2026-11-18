import unittest

import pandas as pd

from fit.t5_memory import pipeline as PIPELINE


class BinocularValidityTests(unittest.TestCase):
    def test_model_input_excludes_device_invalid_sample(self):
        eyetrack = pd.DataFrame(
            {
                "timestamp": [0, 1, 2, 3],
                "bino_eye_gaze_position_x": [960, 900, 1000, 960],
                "bino_eye_gaze_position_y": [540, 500, 550, 540],
                "bino_eye_valid": [1, 0, 1, 1],
                "trigger": [100, 101, 0, 102],
            }
        )
        records = pd.DataFrame(
            {"试次轮次": [1], "刺激图片情绪类型": ["positive"]}
        )

        result = PIPELINE.run_memory_pipeline_from_dataframes(eyetrack, records)["A5"]

        self.assertEqual(result["gaze_x"].tolist(), [960, 1000])
        self.assertNotIn(900, result["gaze_x"].tolist())

    def test_invalid_trigger_sample_does_not_shift_event_time(self):
        eyetrack = pd.DataFrame(
            {
                "timestamp": [0, 1, 2, 3, 4],
                "bino_eye_gaze_position_x": [960, 960, 960, 1000, 960],
                "bino_eye_gaze_position_y": [540, 540, 540, 550, 540],
                "bino_eye_valid": [0, 1, 0, 1, 1],
                "trigger": [100, 0, 101, 0, 102],
            }
        )
        records = pd.DataFrame(
            {"试次轮次": [1], "刺激图片情绪类型": ["positive"]}
        )

        result = PIPELINE.run_memory_pipeline_from_dataframes(eyetrack, records)["A5"]

        fixation_row = result.loc[result["gaze_x"].eq(960)].iloc[0]
        picture_row = result.loc[result["gaze_x"].eq(1000)].iloc[0]
        self.assertEqual(fixation_row["t_from_trial_start"], 1)
        self.assertEqual(picture_row["t_from_trial_start"], 3)
        self.assertEqual(picture_row["t_from_picture_start"], 1)


if __name__ == "__main__":
    unittest.main()
