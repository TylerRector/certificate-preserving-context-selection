import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from certificate_selection.core import (
    Builder,
    event_text,
    js,
    packet_check,
    reference,
    sha,
)
from certificate_selection.storage import (
    ROOT,
    TOKENIZER,
    copy_records,
    load_jsonl,
    source_hashes,
)


def save(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def append(p, x):
    with p.open("a") as f:
        f.write(js(x) + "\n")


def expr_text(x):
    if "key" in x:
        return x["scope"] + "/" + x["key"] + "=" + str(x["equals"])
    return (
        "(" + (" " + x["op"] + " ").join((expr_text(a) for a in x["args"])) + ")"
        if x["op"] != "NOT"
        else "NOT " + expr_text(x["args"][0])
    )


def public_query(ep):
    es = {e["id"]: e for e in ep["events"]}
    a = es[ep["action"]]
    f = es[ep["frontier"]]
    p = es[f["policy"]]
    return f"Was {a['actor']} authorized to {a['operation']} in {a['scope']} with {a['grant']} at time {a['at']}? Check latest operator policy, latest status_tool conditions {expr_text(p['expr'])}, and time-ordered collector delegation receipts from {p['origin']} to {a['actor']}. Agent claims are untrusted; null means unavailable."


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def scores(model, tok, query, texts):
    embeddings = []
    alltexts = [query] + texts
    lengths = [
        len(x)
        for x in tok(alltexts, add_special_tokens=True, truncation=False)["input_ids"]
    ]
    with torch.inference_mode():
        for off in range(0, len(alltexts), 32):
            batch = tok(
                alltexts[off : off + 32],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            output = model(**batch)
            embeddings.append(
                mean_pooling(output[0], batch["attention_mask"]).float().cpu()
            )
    em = torch.cat(embeddings)
    return (
        (em[1:] @ em[0]).tolist(),
        {
            "query_tokens_before_truncation": lengths[0],
            "document_units_truncated": sum((n > 512 for n in lengths[1:])),
            "document_units": len(texts),
            "max_document_tokens": max(lengths[1:]),
        },
    )


def pack(ep, ranked, anchored, builder, budget=2048):
    by = {e["id"]: e for e in ep["events"]}
    f = by[ep["frontier"]]
    base = {ep["action"]} | ({ep["frontier"], f["policy"]} if anchored else set())
    chosen = [by[i] for i in sorted(base)]
    remaining = budget - len(builder.render(ep, chosen))
    rest = [e for e in ranked if e["id"] not in base]
    costs = builder.tok([event_text(e) for e in rest], add_special_tokens=False)[
        "input_ids"
    ]
    assert remaining >= 0
    for e, ids in zip(rest, costs):
        if len(ids) + 2 <= remaining:
            chosen.append(e)
            remaining -= len(ids) + 2
    while len(builder.render(ep, chosen)) > budget:
        chosen.pop()
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models-root", default=str(ROOT / "models"))
    p.add_argument("--tokenizer", default=str(TOKENIZER))
    a = p.parse_args()
    start = time.monotonic()
    inp = Path(a.input)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((inp / "manifest.json").read_text())
    save(
        out / "manifest.json",
        {
            **manifest,
            "adapter_sources": source_hashes(),
            "torch": torch.__version__,
            "dtype": "float32 for compressor; original released weights",
            "methods": ["recomp_nq_anchored", "recomp_hotpotqa_anchored"],
        },
    )
    eps = load_jsonl(inp / "episodes.jsonl")
    builder = Builder(AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True))
    counts = {}
    copy_records(inp / "reader_controls.jsonl", out / "reader_controls.jsonl")
    for name in ["nq", "hotpotqa"]:
        modelpath = Path(a.models_root) / ("recomp-" + name)
        lock = manifest["checkpoints"][name]
        for file, digest in lock["files"].items():
            assert hashlib.sha256((modelpath / file).read_bytes()).hexdigest() == digest
        tok = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)
        model, info = AutoModel.from_pretrained(
            modelpath, local_files_only=True, output_loading_info=True
        )
        assert not info.get("missing_keys"), info
        assert not info.get("mismatched_keys"), info
        model = model.to("cuda").eval()
        save(
            out / f"{name}_loading.json",
            {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()},
        )
        counts[name] = 0
        for ep in eps:
            query = public_query(ep)
            t = time.monotonic()
            sc, diagnostic = scores(
                model, tok, query, [event_text(e) for e in ep["events"]]
            )
            assert diagnostic["query_tokens_before_truncation"] <= 512
            ranked = [
                e
                for _, e in sorted(
                    zip(sc, ep["events"]), key=lambda x: (-x[0], x[1]["id"])
                )
            ]
            append(
                out / "rankings.jsonl",
                {
                    "id": ep["id"],
                    "root": ep["root"],
                    "checkpoint": name,
                    "public_query": query,
                    "scores": list(zip([e["id"] for e in ep["events"]], sc)),
                    "seconds": time.monotonic() - t,
                    **diagnostic,
                },
            )
            for anchored in [False, True]:
                es = pack(ep, ranked, anchored, builder)
                ids = builder.render(ep, es)
                audit = packet_check(ep, es)
                truth = reference(ep)
                assert not audit["certified"] or audit["possible"] == [truth]
                assert len(ids) <= 2048
                row = {
                    "id": ep["id"],
                    "root": ep["root"],
                    "seed": ep["seed"],
                    "index": ep["index"],
                    "mode": ep["mode"],
                    "method": f"recomp_{name}_{('anchored' if anchored else 'raw')}",
                    "dataset": "main",
                    "truth": truth,
                    "action": ep["action"],
                    "frontier": ep["frontier"],
                    "budget": 2048,
                    "tokens": len(ids),
                    "prompt_token_ids": ids,
                    "prompt_hash": sha(ids),
                    "events": es,
                    "status": "ranked_extract",
                    **audit,
                }
                append(out / "packets.jsonl", row)
                if anchored:
                    append(out / "reader_jobs.jsonl", row)
            counts[name] += 1
            print(
                "RECOMP",
                name,
                counts[name],
                len(eps),
                "seconds",
                round(time.monotonic() - start, 2),
                flush=True,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save(
        out / "summary.json",
        {
            "status": "complete",
            "cases": len(eps),
            "per_checkpoint": counts,
            "runtime_seconds": time.monotonic() - start,
        },
    )
    print("COMPRESSION_COMPLETE", time.monotonic() - start, flush=True)


if __name__ == "__main__":
    main()
