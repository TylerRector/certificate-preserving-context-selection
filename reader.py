import argparse
import datetime
import hashlib
import json
import re
import signal
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from core import SEEDS, Builder, js, packet_check, sha
from storage import load_jsonl


def save(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def append(p, x):
    with p.open("a") as f:
        f.write(js(x) + "\n")
        f.flush()


def load(p):
    return load_jsonl(p)


STOP = False


def terminate(*args):
    global STOP
    STOP = True


class End(StoppingCriteria):
    def __init__(self, tok, start, deadline):
        self.tok = tok
        self.start = start
        self.deadline = deadline

    def __call__(self, ids, scores, **kwargs):
        if STOP or time.monotonic() > self.deadline:
            return torch.tensor([True], device=ids.device)
        raw = self.tok.decode(ids[0, self.start :], skip_special_tokens=True)
        return torch.tensor(
            [bool(re.search("(?m)^FINAL:\\s*(ALLOW|VIOLATION|UNKNOWN)\\s*$", raw))],
            device=ids.device,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=int, default=86400)
    a = p.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + a.seconds
    data = Path(a.data)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    b = Builder(tok)
    lock = Path(a.model) / "model_lock.json"
    save(
        out / "manifest.json",
        {
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": json.loads(lock.read_text()) if lock.exists() else a.model,
            "dtype": "bfloat16",
            "torch": torch.__version__,
            "generation": {
                "temperature": 0,
                "do_sample": False,
                "enable_thinking": False,
                "max_new_tokens": 512,
            },
            "data_manifest": json.loads((data / "manifest.json").read_text()),
            "seconds_limit": a.seconds,
            "source_hashes": {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in Path(__file__).parent.glob("*.py")
            },
        },
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.model, local_files_only=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    controls = load(data / "reader_controls.jsonl")
    jobs = load(data / "reader_jobs.jsonl")
    full = load(data / "reader_full_jobs.jsonl")
    cache = {}
    rows = []
    jobs.sort(
        key=lambda r: (
            sha(["reader_order", r["root"]]),
            sha(["method_order", r["root"], r["method"]]),
        )
    )

    def infer(row):
        ep = {
            "action": row["action"],
            "frontier": row["frontier"],
            "arity": row["condition_total"],
        }
        ids = b.render(ep, row["events"])
        assert ids == row["prompt_token_ids"]
        assert len(ids) <= row["budget"]
        key = sha(ids)
        if key not in cache:
            tensor = torch.tensor([ids], device="cuda")
            mask = torch.ones_like(tensor)
            t = time.monotonic()
            torch.cuda.synchronize()
            try:
                with torch.inference_mode():
                    result = model.generate(
                        input_ids=tensor,
                        attention_mask=mask,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        use_cache=True,
                        logits_to_keep=1,
                        stopping_criteria=StoppingCriteriaList(
                            [End(tok, len(ids), deadline)]
                        ),
                    )
                torch.cuda.synchronize()
                output = result[0, len(ids) :].tolist()
                raw = tok.decode(output, skip_special_tokens=True).strip()
                matches = re.findall(
                    "(?m)^FINAL:\\s*(ALLOW|VIOLATION|UNKNOWN)\\s*$", raw
                )
                pred = matches[-1] if matches else "INVALID"
                failure = None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                output = []
                raw = ""
                pred = "INVALID"
                failure = "cuda_out_of_memory"
            cache[key] = {
                "prediction": pred,
                "raw_output": raw,
                "output_token_ids": output,
                "seconds": time.monotonic() - t,
                "output_cap_hit": len(output) >= 512,
                "failure": failure,
            }
            append(
                out / "requests.jsonl",
                {"request_hash": key, "prompt_token_ids": ids, **cache[key]},
            )
        res = cache[key]
        by = {e["id"]: e for e in row["events"]}
        cites = set(re.findall("\\b[eb][0-9a-f]{6,16}\\b", res["raw_output"]))
        fake = sorted(cites - set(by))
        cited = [
            e
            for e in row["events"]
            if e["id"] in cites or e["id"] in (ep["action"], ep["frontier"])
        ]
        cert = packet_check(ep, cited)
        record = {
            k: v for k, v in row.items() if k not in ("events", "prompt_token_ids")
        }
        record.update(res)
        record.update(
            {
                "request_hash": key,
                "cited_ids": sorted(cites),
                "fabricated_ids": fake,
                "citation_certificate": cert["certified"],
                "correct": res["prediction"] == row["truth"],
                "packet_supported_correct": res["prediction"] == row["truth"]
                and row["certified"],
                "grounded_correct": res["prediction"] == row["truth"]
                and cert["certified"]
                and (cert["possible"] == [row["truth"]])
                and (not fake),
                "insufficient_marker": "INSUFFICIENT" in res["raw_output"].upper(),
            }
        )
        append(out / "responses.jsonl", record)
        rows.append(record)

    for i, row in enumerate(controls):
        if STOP or time.monotonic() > deadline - 30:
            break
        infer(row)
    cs = [r for r in rows if r["dataset"] == "control"]
    accuracy = sum((r["correct"] for r in cs)) / max(1, len(cs))
    save(
        out / "qualification.json",
        {
            "planned": len(controls),
            "completed": len(cs),
            "accuracy": accuracy,
            "passed": len(cs) == 12 and accuracy >= 0.9,
            "policy": "Record qualification honestly; do not tune or remove main cases after failures.",
        },
    )
    print("QUALIFICATION", len(cs), accuracy, flush=True)
    for i, row in enumerate(jobs):
        if STOP or time.monotonic() > deadline - 45:
            break
        infer(row)
        if (i + 1) % 12 == 0:
            print(
                "MAIN",
                i + 1,
                len(jobs),
                "seconds",
                round(time.monotonic() - start, 2),
                flush=True,
            )
    per = max([r["seconds"] for r in cache.values()] + [10.0])
    full_run = (
        not STOP and time.monotonic() + max(180.0, len(full) * per * 3) < deadline
    )
    if full_run:
        for row in full:
            if STOP or time.monotonic() > deadline - 30:
                break
            infer(row)
    main = [r for r in rows if r["dataset"] == "main"]
    complete = len(main) == len(jobs)
    metrics = {}
    for method in sorted({r["method"] for r in main}):
        rs = [r for r in main if r["method"] == method]
        metrics[method] = {
            "n": len(rs),
            "accuracy": sum((r["correct"] for r in rs)) / len(rs),
            "packet_supported_accuracy": sum(
                (r["packet_supported_correct"] for r in rs)
            )
            / len(rs),
            "citation_grounded_accuracy": sum((r["grounded_correct"] for r in rs))
            / len(rs),
            "false_approvals": sum(
                (r["truth"] == "VIOLATION" and r["prediction"] == "ALLOW" for r in rs)
            ),
            "violation_n": sum((r["truth"] == "VIOLATION" for r in rs)),
            "invalid": sum((r["prediction"] == "INVALID" for r in rs)),
            "output_cap_hit": sum((r["output_cap_hit"] for r in rs)),
        }
    save(
        out / "summary.json",
        {
            "status": "complete" if complete else "partial_time_limit",
            "planned_main": len(jobs),
            "completed_main": len(main),
            "planned_full": len(full),
            "completed_full": sum((r["dataset"] == "full_source" for r in rows)),
            "full_arm_admitted": full_run,
            "unique_requests": len(cache),
            "runtime_seconds": time.monotonic() - start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "metrics": metrics,
            "per_seed_completed": {
                str(s): sum((r["seed"] == s for r in main)) for s in SEEDS
            },
        },
    )
    print("READER_FINISHED", complete, len(main), time.monotonic() - start, flush=True)


if __name__ == "__main__":
    main()
