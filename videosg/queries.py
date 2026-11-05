"""Eight fixed Cypher queries and a constrained local natural-language router."""

import hashlib
import json
import re
import time
from urllib.request import Request, urlopen

from .pipeline import case_dir, read_json, render, save_json


RELATION = """MATCH (a:VIDEO_Track)-[r:VIDEO_REL]->(b:VIDEO_Track)
WHERE a.run_id=$run_id AND b.run_id=$run_id AND r.run_id=$run_id
AND a.class=$subject_class AND b.class=$object_class AND r.predicate=$predicate
{extra}
RETURN r.rel_uid AS rel_uid, r.start_frame AS start_frame, r.end_frame AS end_frame,
 a.track_uid AS subject, b.track_uid AS object
ORDER BY r.start_frame LIMIT 5"""

PRESENCE = """MATCH (a:VIDEO_Track {run_id:$run_id}) WHERE a.class=$subject_class
RETURN null AS rel_uid, a.first_frame AS start_frame, a.last_frame AS end_frame,
 a.track_uid AS subject, a.track_uid AS object ORDER BY a.first_frame LIMIT 5"""

FOOTAGE_EXISTS = """MATCH (v:VIDEO_Video)<-[:VIDEO_IN_VIDEO]-(a:VIDEO_Track {run_id:$run_id})
WHERE a.class=$subject_class
RETURN null AS rel_uid, a.first_frame AS start_frame, a.last_frame AS end_frame,
 a.track_uid AS subject, a.track_uid AS object ORDER BY a.first_frame LIMIT 1"""

SEQUENCE = """MATCH (a:VIDEO_Track)-[r:VIDEO_REL]->(b:VIDEO_Track),
 (a)-[later:VIDEO_REL]->(b)
WHERE a.run_id=$run_id AND b.run_id=$run_id AND r.run_id=$run_id AND later.run_id=$run_id
 AND a.class=$subject_class AND b.class=$object_class
 AND r.predicate=$first_predicate AND later.predicate=$second_predicate
 AND later.start_s >= r.end_s AND later.start_s-r.end_s <= $within_seconds
RETURN r.rel_uid AS rel_uid, later.rel_uid AS second_rel_uid,
 r.start_frame AS start_frame, later.end_frame AS end_frame,
 a.track_uid AS subject, b.track_uid AS object
ORDER BY r.start_frame LIMIT 5"""

# Parameter names and Cypher text are application-owned. The LLM never supplies Cypher.
TEMPLATES = {
    "presence": (PRESENCE, {"subject_class"}),
    "relation": (RELATION.format(extra=""), {"subject_class", "object_class", "predicate"}),
    "holding": (RELATION.format(extra=""), {"subject_class", "object_class"}),
    "crossing": (RELATION.format(extra=""), {"subject_class", "object_class", "predicate"}),
    "approach": (RELATION.format(extra=""), {"subject_class", "object_class"}),
    "timed_relation": (RELATION.format(extra="AND r.start_s <= $end_s AND r.end_s >= $start_s"),
                       {"subject_class", "object_class", "predicate", "start_s", "end_s"}),
    "sequence": (SEQUENCE, {"subject_class", "object_class", "first_predicate", "second_predicate", "within_seconds"}),
    "footage_exists": (FOOTAGE_EXISTS, {"subject_class"}),
}
TEMPLATE_HELP = {
    "presence": "Intervals spanned by tracks of a specified class.",
    "relation": "Observed relation between two specified classes.",
    "holding": "Person-object co-motion proxy intervals.",
    "crossing": "Enter or leave an explicitly configured region.",
    "approach": "Decreasing separation between two tracks.",
    "timed_relation": "Observed relation overlapping a video-time range in seconds.",
    "sequence": "One relation followed by another within a time limit.",
    "footage_exists": "One representative class-presence result, or NO_MATCH.",
}
PREDICATES = {"near", "touching", "on", "holding", "approaching", "moving_away",
              "entering", "leaving", "lying_on", "falling"}


def prepare(template, params):
    if template not in TEMPLATES:
        raise ValueError("CANNOT_ANSWER: unknown template")
    statement, required = TEMPLATES[template]
    if set(params) != required:
        raise ValueError(f"CANNOT_ANSWER: expected parameters {sorted(required)}")
    params = dict(params)
    for key, value in params.items():
        if key.endswith("_class") or key.endswith("_predicate") or key == "predicate":
            if not isinstance(value, str) or not value or len(value) > 80:
                raise ValueError(f"CANNOT_ANSWER: invalid {key}")
    if template == "holding":
        params["predicate"] = "holding"
    elif template == "approach":
        params["predicate"] = "approaching"
    elif template == "crossing" and params["predicate"] not in {"entering", "leaving"}:
        raise ValueError("CANNOT_ANSWER: crossing requires entering or leaving")
    if any(value not in PREDICATES for key, value in params.items()
           if key in {"predicate", "first_predicate", "second_predicate"}):
        raise ValueError("CANNOT_ANSWER: unsupported predicate")
    if template in {"timed_relation", "sequence"}:
        numeric = ("start_s", "end_s") if template == "timed_relation" else ("within_seconds",)
        for key in numeric:
            if type(params[key]) not in (int, float) or params[key] < 0:
                raise ValueError(f"CANNOT_ANSWER: invalid {key}")
        if template == "timed_relation" and params["start_s"] > params["end_s"]:
            raise ValueError("CANNOT_ANSWER: time range is reversed")
    if not re.match(r"^\s*(MATCH|OPTIONAL MATCH)\b", statement, re.I):
        raise ValueError("Unsafe Cypher template")
    if re.search(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|CALL|APOC)\b|;", statement, re.I):
        raise ValueError("Unsafe Cypher template")
    return statement, params


def route_nl(question, allowed_classes, model="qwen2.5-7b-instruct:latest", endpoint="http://127.0.0.1:11434/api/chat"):
    if re.search(r"\b(delete|drop|merge|create|detach|apoc|dbms|cypher)\b|\b\d{1,2}:\d{2}\b", question, re.I):
        return None
    schema = {name: sorted(required) for name, (_, required) in TEMPLATES.items()}
    prompt = ("You are a strict video-query classifier. Map the user's question to exactly one supported query. Return only JSON with keys "
              "template and params; for unsupported or write requests return {\"template\":\"CANNOT_ANSWER\",\"params\":{}}. "
              "Never invent a class, event or time. Use relation for near, touching, on, lying_on or moving_away; "
              "holding for holds; crossing for entering or leaving; approach for approaching; "
              "timed_relation when a time range is given; sequence for A then B; footage_exists for whether a class appears. "
              "Never choose timed_relation when the question has no explicit start and end seconds. "
              "Do not use zero as a placeholder. Example: 'When is a person near another person?' => "
              "{\"template\":\"relation\",\"params\":{\"subject_class\":\"person\",\"object_class\":\"person\",\"predicate\":\"near\"}}. "
              "Times are seconds. Available classes: " + json.dumps(sorted(allowed_classes))
              + ". Templates and required parameters: " + json.dumps(schema))
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": prompt},
                      {"role": "user", "content": question}], "stream": False, "format": "json"}).encode()
    with urlopen(Request(endpoint, data=body, headers={"Content-Type": "application/json"}), timeout=90) as response:
        try:
            answer = json.loads(json.load(response)["message"]["content"])
        except (KeyError, ValueError, TypeError):
            return None
    if not isinstance(answer, dict) or not isinstance(answer.get("template"), str):
        return None
    if answer.get("template") == "CANNOT_ANSWER":
        return None
    if set(answer) != {"template", "params"} or not isinstance(answer["params"], dict):
        return None
    params = answer["params"]
    # The local model sometimes invents zero-second filters; keep only time bounds present in the question.
    if not re.search(r"\b\d+(?:\.\d+)?\s*(?:s|sec|seconds)\b|\bbetween\s+\d+", question, re.I):
        params.pop("start_s", None)
        params.pop("end_s", None)
        if answer["template"] == "timed_relation":
            answer["template"] = {"holding": "holding", "approach": "approach",
                                  "approaching": "approach", "entering": "crossing",
                                  "leaving": "crossing"}.get(params.get("predicate"), "relation")
    if answer["template"] == "crossing":
        params.pop("start_s", None)
        params.pop("end_s", None)
        if params.get("object_class") not in allowed_classes and f"region:{params.get('object_class')}" in allowed_classes:
            params["object_class"] = f"region:{params['object_class']}"
    if answer["template"] in {"holding", "approach"}:
        params.pop("predicate", None)
    if answer["template"] == "sequence":
        for key in ("first_predicate", "second_predicate"):
            params[key] = {"approach": "approaching", "touch": "touching"}.get(params.get(key), params.get(key))
    if answer["template"] == "presence" or (answer["template"] == "footage_exists" and "footage" not in question.lower()):
        answer["template"] = "footage_exists" if "footage" in question.lower() else "presence"
    try:
        prepare(answer["template"], answer["params"])
    except (ValueError, KeyError, TypeError):
        return None
    if any(value not in allowed_classes for key, value in answer["params"].items()
           if key.endswith("_class")):
        return None
    return answer


def ask(case_id, template, params, uri, user, password, question=None, quiet_no_match=False):
    from neo4j import GraphDatabase, READ_ACCESS

    statement, params = prepare(template, params)
    graph = read_json(case_dir(case_id) / "graph.json")
    bound = {**params, "run_id": graph["run"]["run_id"]}
    driver = GraphDatabase.driver(uri, auth=(user, password))
    started = time.perf_counter()
    try:
        with driver.session(default_access_mode=READ_ACCESS) as session:
            session.run("EXPLAIN " + statement, **bound).consume()
            matches = [dict(row) for row in session.run(statement, **bound)]
    finally:
        driver.close()
    result = {"question": question, "template": template, "cypher": statement,
              "parameters": bound, "matches": matches,
              "query_seconds": round(time.perf_counter() - started, 6)}
    query_id = hashlib.sha256(json.dumps({"question": question, "template": template,
                                        "parameters": bound, "matches": matches}, sort_keys=True).encode()).hexdigest()[:16]
    folder = case_dir(case_id) / "snippets" / query_id
    save_json(folder / "query.json", result)
    if not matches:
        if not quiet_no_match:
            print("NO_MATCH")
        return []
    paths = []
    for index, match in enumerate(matches):
        paths.append(render(case_id, graph, match, f"{query_id}_{index}", result))
        print(paths[-1])
    return paths
