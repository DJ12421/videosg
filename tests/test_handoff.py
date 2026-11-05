import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from videosg.artifacts import rle
from videosg.evaluation import bootstrap_by_video, graph_predictions, score_query, temporal_iou
from videosg.queries import TEMPLATES, TEMPLATE_HELP, prepare
from videosg.rules import region_events
from scripts.meva import activities
from videosg.annotations import consensus
from videosg.pipeline import save_json


class HandoffChecks(unittest.TestCase):
    def test_eight_read_only_templates(self):
        self.assertEqual(len(TEMPLATES), 8)
        self.assertEqual(set(TEMPLATES), set(TEMPLATE_HELP))
        parameters = {
            "presence": {"subject_class": "person"},
            "relation": {"subject_class": "person", "object_class": "person", "predicate": "near"},
            "holding": {"subject_class": "person", "object_class": "backpack"},
            "crossing": {"subject_class": "person", "object_class": "region:door", "predicate": "entering"},
            "approach": {"subject_class": "person", "object_class": "person"},
            "timed_relation": {"subject_class": "person", "object_class": "person", "predicate": "near", "start_s": 1, "end_s": 2},
            "sequence": {"subject_class": "person", "object_class": "person", "first_predicate": "approaching", "second_predicate": "touching", "within_seconds": 5},
            "footage_exists": {"subject_class": "person"},
        }
        for name, params in parameters.items():
            statement, _ = prepare(name, params)
            self.assertIn("MATCH", statement)
            self.assertNotIn("CREATE", statement)

    def test_invalid_query_parameters_cannot_become_cypher(self):
        with self.assertRaises(ValueError):
            prepare("relation", {"subject_class": "person", "object_class": "bag", "predicate": "near", "CREATE": "node"})
        with self.assertRaises(ValueError):
            prepare("crossing", {"subject_class": "person", "object_class": "door", "predicate": "DELETE"})
        with self.assertRaises(ValueError):
            prepare("timed_relation", {"subject_class": "person", "object_class": "person", "predicate": "near", "start_s": 4, "end_s": 2})
        with self.assertRaises(ValueError):
            prepare("relation", {"subject_class": {"$ne": "person"}, "object_class": "bag", "predicate": "near"})

    def test_rle_and_region_crossing(self):
        mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=bool)
        self.assertEqual(rle(mask), [1, 2, 2, 1])
        history = {}
        left = np.array([[1, 0], [0, 0]], dtype=bool)
        right = np.array([[0, 1], [0, 0]], dtype=bool)
        self.assertEqual(region_events(left, {"door": [0.5, 0, 1, 1]}, history, "person", 1), [])
        self.assertEqual(region_events(right, {"door": [0.5, 0, 1, 1]}, history, "person", 2), [("entering", "door")])

    def test_retrieval_and_video_bootstrap(self):
        self.assertAlmostEqual(temporal_iou([0, 2], [1, 3]), 1 / 3)
        self.assertEqual(score_query([], []), {"abstention": 1.0})
        self.assertEqual(score_query([[0, 2]], [[0, 2]])["recall_1_0.7"], 1.0)
        result = bootstrap_by_video([{"video_id": "a", "x": 1}, {"video_id": "b", "x": 0}], "x", repetitions=1000)
        self.assertEqual(result[0], 0.5)

    def test_meva_kpf_metadata_is_not_an_activity(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "clip.activities.yml"
            path.write_text("- {meta: clip was marked empty_37}\n- {act: {act2: {person_talks_to_person: 1.0}, timespan: [{tsr0: [30, 60]}]}}\n", encoding="utf-8")
            self.assertEqual(activities(path), {"person_talks_to_person": [[30, 60]]})

    def test_two_annotator_consensus(self):
        with TemporaryDirectory() as folder:
            first, second, output = (Path(folder) / name for name in ("a.json", "b.json", "consensus.json"))
            event = {"predicate": "near", "subject": "actor_a", "object": "bag", "start_s": 1, "end_s": 3}
            save_json(first, {"annotator_id": "a", "video_sha256": "hash", "events": [event]})
            save_json(second, {"annotator_id": "b", "video_sha256": "hash", "events": [{**event, "start_s": 1.5}]})
            result = consensus(first, second, output)
            self.assertEqual(result["agreement"]["matched"], 1)
            self.assertEqual(result["agreement"]["event_f1"], 1.0)

    def test_oracle_graph_query_uses_relation_interval(self):
        graph = {"video": {"fps": 10}, "tracks": [
            {"track_uid": "a", "class": "person", "first_frame": 0, "last_frame": 99},
            {"track_uid": "b", "class": "backpack", "first_frame": 0, "last_frame": 99}],
            "relations": [{"subject": "a", "object": "b", "predicate": "holding", "start_s": 2, "end_s": 4}]}
        self.assertEqual(graph_predictions(graph, {"subject_class": "person", "object_class": "backpack",
                                                   "predicate": "holding"}), [[2, 4]])


if __name__ == "__main__":
    unittest.main()
