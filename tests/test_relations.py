import unittest

from videosg.pipeline import aggregate


class RelationIntervalTest(unittest.TestCase):
    def test_gap_merge_and_source_frames(self):
        observations = [
            {"subject": "a", "predicate": "near", "object": "b", "frame": frame}
            for frame in (0, 10, 30)
        ]
        result = aggregate(observations, step=10, last_frame=35, gap_samples=0, min_support=1, run_id="run", fps=10)
        self.assertEqual([(r["start_frame"], r["end_frame"]) for r in result], [(0, 19), (30, 35)])
        self.assertEqual(result[0]["support_frames"], [0, 10])


if __name__ == "__main__":
    unittest.main()
