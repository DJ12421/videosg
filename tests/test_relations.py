import unittest

from videosg.pipeline import aggregate


class RelationIntervalTest(unittest.TestCase):
    def test_gap_merge_and_source_frames(self):
        observations = [
            {"subject": "a", "predicate": "near", "object": "b", "frame": frame, "score": score}
            for frame, score in ((0, 0.8), (10, 0.6), (30, 0.7))
        ]
        result = aggregate(observations, step=10, last_frame=35, gap_samples=0, min_support=1, run_id="run", fps=10)
        self.assertEqual([(r["start_frame"], r["end_frame"]) for r in result], [(0, 19), (30, 35)])
        self.assertEqual(result[0]["support_frames"], [0, 10])
        self.assertAlmostEqual(result[0]["interval_conf"], 0.7)

    def test_gap_merge_penalizes_missing_support(self):
        observations = [{"subject": "a", "predicate": "near", "object": "b", "frame": frame, "score": 0.9}
                        for frame in (0, 20)]
        result = aggregate(observations, step=10, last_frame=29, gap_samples=1,
                           min_support=1, run_id="run", fps=10)
        self.assertAlmostEqual(result[0]["interval_conf"], 0.6)


if __name__ == "__main__":
    unittest.main()
