"""Plot observed relation intervals from the traceable public demo graph."""

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from videosg.pipeline import ROOT, read_json


def main():
    graph = read_json(ROOT / "cases" / "demo" / "video" / "graph.json")
    counts = Counter(row["predicate"] for row in graph["relations"])
    predicates = [name for name, _ in counts.most_common()]
    figure, axis = plt.subplots(figsize=(12, max(3, len(predicates) * 0.65)))
    for row in graph["relations"]:
        y = predicates.index(row["predicate"])
        axis.broken_barh([(row["start_s"], row["end_s"] - row["start_s"])],
                         (y - 0.28, 0.56), alpha=0.2, color="#145a8d")
    axis.set_yticks(range(len(predicates)), [f"{name} ({counts[name]})" for name in predicates])
    axis.set_xlim(0, graph["video"]["duration_s"])
    axis.set_xlabel("Seconds from first source frame")
    axis.set_title(f"Observed rule intervals | {graph['video']['video_id']} | run {graph['run']['run_id']}")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output = ROOT / "eval" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "demo_timeline.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
