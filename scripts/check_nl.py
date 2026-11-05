"""Exercise the local Ollama router on the handoff's 20 questions."""

import json
from pathlib import Path

from videosg.queries import route_nl


cases = json.loads((Path(__file__).parents[1] / "tests" / "nl_questions.json").read_text(encoding="utf-8"))
rows = []
for case in cases:
    result = route_nl(case["question"], {"person", "backpack", "region:door"})
    actual = result["template"] if result else "CANNOT_ANSWER"
    rows.append({**case, "actual": actual, "pass": actual == case["expected"]})
    print(f"{case['kind']}: {actual} expected {case['expected']}", flush=True)
output = Path(__file__).parents[1] / "eval" / "nl_questions.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
for kind in ("answerable", "unanswerable", "adversarial"):
    group = [row for row in rows if row["kind"] == kind]
    print(f"{kind}: {sum(row['pass'] for row in group)}/{len(group)}")
