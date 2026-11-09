import argparse
import hashlib
import json
from pathlib import Path

from core import SEEDS, sha
from storage import ROOT, append, copy_records, iter_jsonl, load_jsonl, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, default=ROOT / "data/final")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/recomp/input")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = load_jsonl(args.final / "cpu/reader_jobs.jsonl")
    roots = set()
    for seed in SEEDS:
        candidates = {row["root"] for row in jobs if row["seed"] == seed}
        roots.update(
            sorted(candidates, key=lambda root: sha(["recomp-preliminary-v1", root]))[
                :2
            ]
        )
    episodes = [
        ep
        for ep in iter_jsonl(args.final / "cpu/episodes.jsonl")
        if ep["root"] in roots and ep["mode"] == "primary"
    ]
    assert len(episodes) == 8
    for episode in sorted(episodes, key=lambda ep: ep["root"]):
        append(args.out / "episodes.jsonl", episode)
    copy_records(
        args.final / "cpu/reader_controls.jsonl", args.out / "reader_controls.jsonl"
    )
    for row in iter_jsonl(args.final / "32b/responses.jsonl"):
        if row["dataset"] == "main" and row["root"] in roots:
            append(args.out / "existing_32b.jsonl", row)
    models = json.loads((ROOT / "data/models.json").read_text())
    save(
        args.out / "manifest.json",
        {
            "study": "Preliminary RECOMP transfer comparison",
            "roots": sorted(roots),
            "seeds": SEEDS,
            "cases": 8,
            "source_sha256": hashlib.sha256(
                (args.out / "episodes.jsonl").read_bytes()
            ).hexdigest(),
            "checkpoints": {
                name: models["recomp-" + name] for name in ["nq", "hotpotqa"]
            },
            "upstream_repo": "https://github.com/carriex/recomp",
            "upstream_commit": "51d4432151efb3275257a9407dc71d1e5ec6634d",
            "inference": "Unchanged released extractive weights; whole-event units; mean pooling and dot products",
            "arms": "Action-only anchor and shared action/policy/snapshot anchors; reader evaluates shared-header arms",
        },
    )


if __name__ == "__main__":
    main()
