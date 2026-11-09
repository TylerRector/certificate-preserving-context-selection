import argparse
import copy
import datetime
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from certificate_selection.core import (
    METHODS,
    SEEDS,
    Builder,
    generate,
    packet_check,
    reference,
    sha,
)
from certificate_selection.dataset import integrity, packet, source, tests
from certificate_selection.storage import ROOT, TOKENIZER, append, save, source_hashes

WORK_BUILDER = None
WORK_METHODS = None


def initialize_worker(tokenizer, methods):
    global WORK_BUILDER, WORK_METHODS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    WORK_BUILDER = Builder(
        AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    )
    WORK_METHODS = methods


def root_work(job):
    seed, i, is_long = job
    b = WORK_BUILDER
    methods = WORK_METHODS
    primary = source(generate(seed, i, background=0), b, 16384)
    integrity(primary)
    raw_a = generate(seed, i, mode="edge_allow", background=0)
    raw_v = generate(seed, i, mode="edge_deny", background=0)
    changes = [(x, y) for x, y in zip(raw_a["events"], raw_v["events"]) if x != y]
    assert len(changes) == 1 and changes[0][0]["kind"] == "receipt"
    edge_a = source(raw_a, b, 16384)
    edge_v = copy.deepcopy(edge_a)
    edge_v["id"] = raw_v["id"]
    edge_v["mode"] = raw_v["mode"]
    edge_v["events"] = [
        copy.deepcopy(changes[0][1]) if e["id"] == changes[0][0]["id"] else e
        for e in edge_v["events"]
    ]
    edge_v["source_tokens"] = len(b.render(edge_v, edge_v["events"]))
    integrity(edge_a)
    integrity(edge_v)
    assert reference(edge_a) == "ALLOW" and reference(edge_v) == "VIOLATION"
    assert (
        len(edge_a["events"]) == len(edge_v["events"])
        and sum((x != y for x, y in zip(edge_a["events"], edge_v["events"]))) == 1
    )
    rows = [packet(ep, m, 2048, b) for ep in [primary, edge_a, edge_v] for m in methods]
    long = []
    minimal = []
    if i < 8:
        row = next(
            (
                r
                for r in rows
                if r["mode"] == "primary" and r["method"] == "multi_candidate"
            )
        )
        essential = sum(
            (
                not packet_check(
                    primary, [x for x in row["events"] if x["id"] != e["id"]]
                )["certified"]
                for e in row["events"]
            )
        )
        minimal.append(
            {
                "seed": seed,
                "index": i,
                "certified": row["certified"],
                "records": len(row["events"]),
                "individually_essential": essential,
            }
        )
    if is_long:
        for length in [16384, 65536, 114688]:
            ep = (
                primary
                if length == 16384
                else source(generate(seed, i, background=0), b, length)
            )
            assert reference(ep) == reference(primary)
            for m in METHODS:
                for budget in [1024, 2048, 4096]:
                    r = packet(ep, m, budget, b)
                    r["source_tokens"] = ep["source_tokens"]
                    r["position"] = ep["position"]
                    long.append(r)
    return ([primary, edge_a, edge_v], rows, long, minimal)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    p.add_argument("--out", default=str(ROOT / "runs/final/cpu"))
    p.add_argument("--per-seed", type=int, default=512)
    p.add_argument("--long-roots", type=int, default=96)
    p.add_argument("--reader-per-seed", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--test-only", action="store_true")
    a = p.parse_args()
    started = time.monotonic()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    b = Builder(tok)
    check = tests(b)
    print("VALIDATION", check, "roots", flush=True)
    if a.test_only:
        return
    dev = []
    for seed in SEEDS:
        for i in range(8):
            ep = generate(seed, i, "dev", background=0)
            for m in ["certificate_greedy", "dependency_closure"]:
                dev.append(packet(ep, m, 2048, b))
    scores = {
        m: {
            "certified": sum((r["certified"] for r in dev if r["method"] == m)),
            "tokens": sum((r["tokens"] for r in dev if r["method"] == m)),
        }
        for m in ["certificate_greedy", "dependency_closure"]
    }
    best = min(scores, key=lambda m: (-scores[m]["certified"], scores[m]["tokens"], m))
    methods = ["multi_candidate", best, "okapi_bm25"]
    save(
        out / "baseline_selection.json",
        {"split": "dev", "roots": 32, "scores": scores, "selected": best},
    )
    plan = {
        "version": "certificate-preserving-context-selection-v1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seeds": SEEDS,
        "per_seed": a.per_seed,
        "primary_roots": 4 * a.per_seed,
        "reader_per_seed": a.reader_per_seed,
        "primary_source_tokens": 16384,
        "primary_packet_budget": 2048,
        "methods": methods,
        "strong_baseline": best,
        "long_roots": a.long_roots,
        "data_randomness": "Four independent fixed seeds; root-specific hash streams; deterministic greedy model decoding",
        "primary_tests": ["certificate_vs_strong", "causal_pair_certificate_vs_strong"],
        "alpha_each": 0.025,
        "source_hashes": source_hashes(),
        "tokenizer_template_hash": sha(tok.chat_template),
        "trust": "Authenticated collector snapshots/receipts; no resistance to collector compromise",
        "reader_status": "exploratory",
        "source_format": "Readable JSON events with explicit binding fields; collector frontier counts toward packet cost",
    }
    save(out / "manifest.json", plan)
    order = [(seed, i) for i in range(a.per_seed) for seed in SEEDS]
    reader = {
        (seed, i)
        for seed in SEEDS
        for i in sorted(range(a.per_seed), key=lambda j: sha(["reader", seed, j]))[
            : a.reader_per_seed
        ]
    }
    longset = set(sorted(order, key=lambda x: sha(["long", *x]))[: a.long_roots])
    completed = 0
    reader_eps = []
    stats = []
    minimal = []
    global WORK_BUILDER, WORK_METHODS
    WORK_BUILDER = b
    WORK_METHODS = methods
    with ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(a.tokenizer, methods),
    ) as pool:
        results = pool.map(
            root_work, [(s, i, (s, i) in longset) for s, i in order], chunksize=1
        )
        for (seed, i), (episodes, rows, long, mins) in zip(order, results):
            primary = episodes[0]
            for ep in episodes:
                append(out / "episodes.jsonl", ep)
            for row in rows:
                append(out / "packets.jsonl", row)
                stats.append(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("events", "prompt_token_ids")
                    }
                )
            minimal.extend(mins)
            if (seed, i) in reader:
                reader_eps.append(primary)
                for r in rows:
                    if r["mode"] == "primary":
                        append(
                            out / "reader_jobs.jsonl",
                            {
                                **r,
                                "dataset": "main",
                                "action": primary["action"],
                                "frontier": primary["frontier"],
                            },
                        )
            for r in long:
                append(out / "long_packets.jsonl", r)
            completed += 1
            if completed % 16 == 0:
                print(
                    "CPU_ROOTS",
                    completed,
                    len(order),
                    "seconds",
                    round(time.monotonic() - started, 2),
                    flush=True,
                )
    for seed in SEEDS:
        for ep in sorted(
            (e for e in reader_eps if e["seed"] == seed),
            key=lambda x: sha(["full", x["root"]]),
        )[:2]:
            ids = b.render(ep, ep["events"])
            append(
                out / "reader_full_jobs.jsonl",
                {
                    "id": ep["id"],
                    "root": ep["root"],
                    "seed": seed,
                    "index": ep["index"],
                    "mode": "primary",
                    "truth": reference(ep),
                    "method": "full",
                    "dataset": "full_source",
                    "action": ep["action"],
                    "frontier": ep["frontier"],
                    "events": ep["events"],
                    "prompt_token_ids": ids,
                    "prompt_hash": sha(ids),
                    "tokens": len(ids),
                    "budget": 20000,
                    **packet_check(ep, ep["events"]),
                },
            )
    controls = []
    for seed in SEEDS:
        seen = set()
        for i in range(100):
            ep = generate(seed, i, "qualification", background=0)
            if ep["arity"] != 2:
                continue
            r = packet(ep, "multi_candidate", 2048, b)
            if r["certified"] and r["truth"] not in seen:
                controls.append(
                    {
                        **r,
                        "dataset": "control",
                        "action": ep["action"],
                        "frontier": ep["frontier"],
                    }
                )
                seen.add(r["truth"])
            if len(seen) == 3:
                break
        assert len(seen) == 3
    for r in controls:
        append(out / "reader_controls.jsonl", r)
    save(out / "minimality.json", minimal)
    save(
        out / "cpu_summary.json",
        {
            "status": "complete" if completed == len(order) else "partial_time_limit",
            "completed_roots": completed,
            "planned_roots": len(order),
            "runtime_seconds": time.monotonic() - started,
            "reader_roots": len(reader_eps),
            "per_seed_completed": {
                str(s): sum(
                    (
                        r["seed"] == s
                        and r["method"] == "multi_candidate"
                        and (r["mode"] == "primary")
                        for r in stats
                    )
                )
                for s in SEEDS
            },
        },
    )
    save(out / "cpu_metrics.json", stats)
    print("CPU_FINISHED", completed, "runtime", time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()
