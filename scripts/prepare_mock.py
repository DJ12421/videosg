"""Build an oracle graph and evaluation manifest from cleared, reconciled mock labels."""

from pathlib import Path

from videosg.pipeline import ROOT, case_dir, read_json, save_json


def prepare(scene_path, consensus_path):
    scene, consensus = read_json(Path(scene_path)), read_json(Path(consensus_path))
    if not scene["clearance_confirmed"] or not scene["consent_forms_outside_repo"]:
        raise ValueError("Mock footage requires confirmed clearance and off-repo consent forms")
    case_id = scene["case_id"]
    graph = read_json(case_dir(case_id) / "graph.json")
    video = graph["video"]
    if video["sha256"] != consensus["video_sha256"]:
        raise ValueError("Consensus and video hashes differ")
    objects = {name: "person" for name in scene["actors"]}
    objects.update({name: name for name in scene["props"]})
    objects.update({name: f"region:{name}" for name in scene.get("regions", [])})
    tracks = [{"track_uid": f"oracle:{name}", "class": label,
               "first_frame": 0, "last_frame": video["n_frames"] - 1}
              for name, label in objects.items()]
    relations = []
    for index, event in enumerate(consensus["events"]):
        if event["subject"] not in objects or event["object"] not in objects:
            raise ValueError(f"Unknown annotated actor or object in event {index}")
        relations.append({"rel_uid": f"oracle:{index}", "subject": f"oracle:{event['subject']}",
                          "object": f"oracle:{event['object']}", "predicate": event["predicate"],
                          "start_s": event["start_s"], "end_s": event["end_s"], "score": 1.0})
    destination = ROOT / "data" / "mock" / case_id
    oracle = destination / "oracle_graph.json"
    save_json(oracle, {"video": video, "tracks": tracks, "relations": relations})
    groups = {}
    for event in consensus["events"]:
        key = (event["predicate"], objects[event["subject"]], objects[event["object"]])
        groups.setdefault(key, []).append([event["start_s"], event["end_s"]])
    queries = [{"text": f"When is a {subject} {predicate} a {obj}?", "predicate": predicate,
                "subject_class": subject, "object_class": obj, "truth": spans,
                "typical_duration_s": 4.0}
               for (predicate, subject, obj), spans in groups.items()]
    manifest = destination / "evaluation_manifest.json"
    save_json(manifest, [{"dataset": "mock", "case_id": case_id, "oracle_graph": str(oracle),
                          "queries": queries}])
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("consensus")
    args = parser.parse_args()
    print(prepare(args.scene, args.consensus))
