"""Make one Neo4j-backed, provenance-checked evidence snippet per evaluated video."""

import json
import os

from neo4j import GraphDatabase, READ_ACCESS

from videosg.audit import provenance
from videosg.pipeline import ROOT, case_dir, read_json, render


def main():
    base = ROOT / "data" / "meva"
    environment = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
    uri = os.getenv("NEO4J_URI", environment.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    user = os.getenv("NEO4J_USER", environment.get("NEO4J_USER", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD") or environment.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("Neo4j password required for evidence samples")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    report = []
    entries = read_json(base / "evaluation_manifest.json")
    try:
        with driver.session(default_access_mode=READ_ACCESS) as session:
            for entry in entries:
                case_id = entry["case_id"]
                graph = read_json(case_dir(case_id) / "graph.json")
                statement = """MATCH (a:VIDEO_Track)-[r:VIDEO_REL]->(b:VIDEO_Track)
WHERE r.run_id=$run_id AND a.run_id=$run_id AND b.run_id=$run_id
RETURN r.rel_uid AS rel_uid, r.start_frame AS start_frame, r.end_frame AS end_frame,
 a.track_uid AS subject, b.track_uid AS object
ORDER BY r.end_frame-r.start_frame ASC LIMIT 1"""
                session.run("EXPLAIN " + statement, run_id=graph["run"]["run_id"]).consume()
                match = session.run(statement, run_id=graph["run"]["run_id"]).single()
                if match is None:
                    report.append({"case_id": case_id, "status": "NO_MATCH"})
                    continue
                mp4 = render(case_id, graph, dict(match), "meva_evidence_sample",
                             {"template": "relation", "cypher": statement, "run_id": graph["run"]["run_id"]})
                audit = provenance(case_id, uri, user, password)
                report.append({"case_id": case_id, "status": "ok", "snippet": str(mp4),
                               "sidecars_checked": audit["sidecars_checked"]})
                print(f"Evidence {len(report)}/{len(entries)}: {case_id}", flush=True)
    finally:
        driver.close()
    destination = ROOT / "eval" / "meva_evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
