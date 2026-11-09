import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from certificate_selection.audit import audit_reader
from certificate_selection.core import Builder, packet_check, reference
from certificate_selection.storage import ROOT, TOKENIZER, load_jsonl, open_text


def load(p):
    return load_jsonl(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", default=str(ROOT / "data/recomp"))
    p.add_argument("--final", default=str(ROOT / "data/final"))
    p.add_argument("--output", default=str(ROOT / "outputs/recomp"))
    a = p.parse_args()
    root = Path(a.result)
    inp = root / "input"
    eps = {e["id"]: e for e in load(inp / "episodes.jsonl")}
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    builder = Builder(tok)
    packets = load(root / "compression/packets.jsonl")
    new = load(root / "32b/responses.jsonl")
    old = load(inp / "existing_32b.jsonl")
    response_checks = audit_reader(
        new,
        packets + load(root / "compression/reader_controls.jsonl"),
        eps,
        builder,
        load(root / "32b/requests.jsonl"),
    )
    oldpackets = []
    with open_text(Path(a.final) / "cpu/packets.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in eps and r["mode"] == "primary":
                oldpackets.append(r)
    for r in packets:
        ep = eps[r["id"]]
        by = {e["id"]: e for e in ep["events"]}
        assert all((by[e["id"]] == e for e in r["events"]))
        assert builder.render(ep, r["events"]) == r["prompt_token_ids"]
        assert len(r["prompt_token_ids"]) <= 2048
        assert packet_check(ep, r["events"])["certified"] == r["certified"]
        assert not r["certified"] or r["possible"] == [reference(ep)]
    methods = sorted({r["method"] for r in packets + oldpackets})
    comparison = {}
    for m in methods:
        rs = [r for r in packets + oldpackets if r["method"] == m]
        rr = [r for r in old + new if r["method"] == m and r["dataset"] == "main"]
        vi = [r for r in rr if r["truth"] == "VIOLATION"]
        comparison[m] = {
            "cases": len(rs),
            "certificate_recovery": sum((r["certified"] for r in rs)),
            "condition_kept": sum((r["condition_kept"] for r in rs)),
            "condition_total": sum((r["condition_total"] for r in rs)),
            "mean_tokens": sum((r["tokens"] for r in rs)) / max(1, len(rs)),
            "reader_cases": len(rr),
            "reader_correct": sum((r["correct"] for r in rr)),
            "reader_grounded": sum((r["grounded_correct"] for r in rr)),
            "false_approvals": sum((r["prediction"] == "ALLOW" for r in vi)),
            "violations": len(vi),
            "reader_invalid": sum((r["prediction"] == "INVALID" for r in rr)),
            "reader_truncated": sum((r["output_cap_hit"] for r in rr)),
        }
    ranks = load(root / "compression/rankings.jsonl")
    result = {
        "reader_response_checks": response_checks,
        "status": "complete"
        if len(packets) == 32 and sum((r["dataset"] == "main" for r in new)) == 16
        else "partial",
        "sample": json.loads((inp / "manifest.json").read_text()),
        "comparison": comparison,
        "ranking_truncation": {
            name: {
                "queries_truncated": sum(
                    (
                        r["query_tokens_before_truncation"] > 512
                        for r in ranks
                        if r["checkpoint"] == name
                    )
                ),
                "document_units_truncated": sum(
                    (
                        r["document_units_truncated"]
                        for r in ranks
                        if r["checkpoint"] == name
                    )
                ),
                "document_units": sum(
                    (r["document_units"] for r in ranks if r["checkpoint"] == name)
                ),
            }
            for name in ["nq", "hotpotqa"]
        },
        "new_reader_qualification": json.loads(
            (root / "32b/qualification.json").read_text()
        )
        if (root / "32b/qualification.json").exists()
        else {},
        "primary_claim": "Exploratory external-method transfer feasibility on eight existing cases; no significance or general superiority claim.",
    }
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RECOMP", result["status"], flush=True)


if __name__ == "__main__":
    main()
