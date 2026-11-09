import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest
from transformers import AutoTokenizer

from certificate_selection.audit import audit_reader
from certificate_selection.core import (
    SEEDS,
    Builder,
    generate,
    packet_check,
    reference,
    sha,
)
from certificate_selection.dataset import integrity, source
from certificate_selection.storage import ROOT, TOKENIZER, load_jsonl, open_text


def load(p):
    return load_jsonl(p)


def paired(a, b):
    wins = sum((x and (not y) for x, y in zip(a, b)))
    loss = sum((y and (not x) for x, y in zip(a, b)))
    n = len(a)
    return {
        "n": n,
        "wins": wins,
        "losses": loss,
        "difference": (sum(a) - sum(b)) / max(1, n),
        "two_sided_p": binomtest(wins, wins + loss, 0.5).pvalue if wins + loss else 1.0,
        "alpha": 0.025,
        "reject": wins > loss
        and (binomtest(wins, wins + loss, 0.5).pvalue if wins + loss else 1.0) < 0.025,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT / "data/final"))
    p.add_argument("--output", default=str(ROOT / "outputs/final"))
    p.add_argument("--tokenizer")
    a = p.parse_args()
    root = Path(a.root)
    data = root / "cpu"
    manifest = json.loads((data / "manifest.json").read_text())
    rows = json.loads((data / "cpu_metrics.json").read_text())
    best = manifest["strong_baseline"]
    eps = {e["id"]: e for e in load(data / "episodes.jsonl")}
    sourcechecks = 0
    certchecks = 0
    tokenpath = a.tokenizer or str(TOKENIZER)
    tok = AutoTokenizer.from_pretrained(tokenpath, local_files_only=True)
    builder = Builder(tok)
    for ep in eps.values():
        integrity(ep)
    for line in open_text(data / "packets.jsonl"):
        row = json.loads(line)
        ep = eps[row["id"]]
        original = {e["id"]: e for e in ep["events"]}
        assert all((original[e["id"]] == e for e in row["events"]))
        sourcechecks += len(row["events"])
        r = packet_check(ep, row["events"])
        assert r["certified"] == row["certified"] and r["possible"] == row["possible"]
        assert not r["certified"] or r["possible"] == [reference(ep)]
        assert sha(row["prompt_token_ids"]) == row["prompt_hash"]
        assert builder.render(ep, row["events"]) == row["prompt_token_ids"]
        assert len(row["prompt_token_ids"]) == row["tokens"] <= row["budget"]
        certchecks += 1
    by = {(r["root"], r["mode"], r["method"]): r for r in rows}
    roots = sorted({r["root"] for r in rows})
    tests = {}
    for name, modes in [
        ("primary", ["primary"]),
        ("causal_pair", ["edge_allow", "edge_deny"]),
    ]:
        complete = [
            r
            for r in roots
            if all(
                (
                    (r, mode, m) in by
                    for mode in modes
                    for m in ["multi_candidate", best]
                )
            )
        ]
        xs = [
            all((by[r, mode, "multi_candidate"]["certified"] for mode in modes))
            for r in complete
        ]
        ys = [all((by[r, mode, best]["certified"] for mode in modes)) for r in complete]
        tests[name] = paired(xs, ys)
    groups = defaultdict(list)
    for r in rows:
        groups[r["method"], r["prompt_hash"]].append(r)
    conflicting = {
        key for key, rs in groups.items() if len({r["truth"] for r in rs}) > 1
    }
    verified = {}
    for line in open_text(data / "packets.jsonl"):
        r = json.loads(line)
        key = (r["method"], r["prompt_hash"])
        if key in conflicting:
            if key in verified:
                assert verified[key] == r["prompt_token_ids"]
            else:
                verified[key] = r["prompt_token_ids"]
    collisions = {}
    for method in manifest["methods"]:
        gs = [rs for (m, h), rs in groups.items() if m == method]
        bad = [rs for rs in gs if len({r["truth"] for r in rs}) > 1]
        n = sum(map(len, gs))
        ceiling = sum(
            (max(Counter((r["truth"] for r in rs)).values()) for rs in gs)
        ) / max(1, n)
        collisions[method] = {
            "n_related_worlds": n,
            "conflicting_groups": len(bad),
            "affected_worlds": sum(map(len, bad)),
            "dataset_specific_packet_accuracy_ceiling": ceiling,
            "exact_arrays_verified": True,
        }
    cpu = {}
    for seed in ["all"] + SEEDS:
        cpu[str(seed)] = {}
        for method in manifest["methods"]:
            rs = [
                r
                for r in rows
                if r["mode"] == "primary"
                and r["method"] == method
                and (seed == "all" or r["seed"] == seed)
            ]
            cpu[str(seed)][method] = {
                "n": len(rs),
                "certified": sum((r["certified"] for r in rs)),
                "condition_kept": sum((r["condition_kept"] for r in rs)),
                "condition_total": sum((r["condition_total"] for r in rs)),
                "mean_tokens": sum((r["tokens"] for r in rs)) / max(1, len(rs)),
                "overflow": sum((r["status"] == "overflow" for r in rs)),
            }
    long = defaultdict(list)
    long_sources = {}
    longchecks = 0
    for r in load(data / "long_packets.jsonl"):
        target = min([16384, 65536, 114688], key=lambda x: abs(x - r["source_tokens"]))
        key = (r["root"], target)
        if key not in long_sources:
            long_sources[key] = source(
                generate(r["seed"], r["index"], background=0), builder, target
            )
        ep = long_sources[key]
        original = {e["id"]: e for e in ep["events"]}
        assert all((original[e["id"]] == e for e in r["events"]))
        assert builder.render(ep, r["events"]) == r["prompt_token_ids"]
        assert len(r["prompt_token_ids"]) <= r["budget"]
        assert packet_check(ep, r["events"])["certified"] == r["certified"]
        longchecks += 1
        long[r["method"], r["budget"], target].append(r)
    longmetrics = [
        {
            "method": m,
            "budget": budget,
            "source_target": length,
            "n": len(rs),
            "certified": sum((r["certified"] for r in rs)),
            "condition_fraction": sum((r["condition_kept"] for r in rs))
            / max(1, sum((r["condition_total"] for r in rs))),
        }
        for (m, budget, length), rs in sorted(long.items())
    ]
    readers = {}
    reader_jobs = (
        load(data / "reader_controls.jsonl")
        + load(data / "reader_jobs.jsonl")
        + load(data / "reader_full_jobs.jsonl")
    )
    for name in ["14b", "32b"]:
        path = root / name
        res = load(path / "responses.jsonl")
        response_checks = audit_reader(
            res, reader_jobs, eps, builder, load(path / "requests.jsonl")
        )
        mainrows = [r for r in res if r["dataset"] == "main"]
        metrics = {}
        for seed in ["all"] + SEEDS:
            metrics[str(seed)] = {}
            for method in manifest["methods"]:
                rs = [
                    r
                    for r in mainrows
                    if r["method"] == method and (seed == "all" or r["seed"] == seed)
                ]
                vi = [r for r in rs if r["truth"] == "VIOLATION"]
                fa = sum((r["prediction"] == "ALLOW" for r in vi))
                up = binomtest(fa, len(vi)).proportion_ci(0.95).high if vi else None
                metrics[str(seed)][method] = {
                    "n": len(rs),
                    "correct": sum((r["correct"] for r in rs)),
                    "packet_supported_correct": sum(
                        (r["packet_supported_correct"] for r in rs)
                    ),
                    "citation_grounded_correct": sum(
                        (r["grounded_correct"] for r in rs)
                    ),
                    "violations": len(vi),
                    "false_approvals": fa,
                    "false_approval_exact_95ci_high": up,
                    "invalid": sum((r["prediction"] == "INVALID" for r in rs)),
                    "truncated": sum((r["output_cap_hit"] for r in rs)),
                }
        readers[name] = {
            "response_checks": response_checks,
            "summary": json.loads((path / "summary.json").read_text())
            if (path / "summary.json").exists()
            else {"status": "missing_or_interrupted"},
            "qualification": json.loads((path / "qualification.json").read_text())
            if (path / "qualification.json").exists()
            else {},
            "metrics": metrics,
        }
    result = {
        "manifest": manifest,
        "source_record_checks": sourcechecks,
        "packet_certificate_checks": certchecks,
        "long_packet_checks": longchecks,
        "completed_roots": len(roots),
        "planned_roots": manifest["primary_roots"],
        "cpu": cpu,
        "primary_tests": tests,
        "collisions": collisions,
        "long_context": longmetrics,
        "readers": readers,
        "statistical_note": "Two corrected CPU tests; reader comparisons exploratory. Four fixed seeds are strata, not a claim of broad domain generalization.",
        "audit_note": "Exact source records, set-valued checker, native retokenization, token hashes and budgets audited.",
    }
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AUDITED", len(roots), certchecks, flush=True)


if __name__ == "__main__":
    main()
