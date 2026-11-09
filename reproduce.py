import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from storage import ROOT

LABELS = {
    "multi_candidate": "Multi-candidate certificate selection",
    "certificate_greedy": "Certificate-greedy selection",
    "okapi_bm25": "Okapi BM25",
    "recomp_nq_anchored": "RECOMP Natural Questions, anchored",
    "recomp_hotpotqa_anchored": "RECOMP HotpotQA, anchored",
}


def csv_table(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_table(path, headers, rows):
    def escape(value):
        return str(value).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")

    lines = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\hline"]
    lines.append(" & ".join(map(escape, headers)) + r" \\")
    lines.append(r"\hline")
    lines.extend(" & ".join(map(escape, row)) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, default=ROOT / "data/final")
    parser.add_argument("--recomp", type=Path, default=ROOT / "data/recomp")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, USE_TORCH="0", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    commands = [
        ["analyze.py", "--root", str(args.final), "--output", str(args.out / "final")],
        [
            "analyze_recomp.py",
            "--result",
            str(args.recomp),
            "--final",
            str(args.final),
            "--output",
            str(args.out / "recomp"),
        ],
    ]
    for command in commands:
        subprocess.run(
            [sys.executable, str(ROOT / command[0]), *command[1:]], env=env, check=True
        )
    final = json.loads((args.out / "final/analysis.json").read_text())
    recomp = json.loads((args.out / "recomp/analysis.json").read_text())
    table1 = []
    for method in ["multi_candidate", "certificate_greedy", "okapi_bm25"]:
        row = final["cpu"]["all"][method]
        table1.append(
            {
                "Selector": LABELS[method],
                "Certified": f"{row['certified']}/{row['n']}",
                "Conditions": f"{row['condition_kept']}/{row['condition_total']}",
                "Mean tokens": f"{row['mean_tokens']:.1f}",
            }
        )
    table2 = []
    for method in [
        "multi_candidate",
        "certificate_greedy",
        "recomp_nq_anchored",
        "recomp_hotpotqa_anchored",
        "okapi_bm25",
    ]:
        row = recomp["comparison"][method]
        table2.append(
            {
                "Method": LABELS[method],
                "Certified evidence": f"{row['certificate_recovery']}/{row['cases']}",
                "Mean tokens": f"{row['mean_tokens']:.1f}",
                "32B correct": f"{row['reader_correct']}/{row['reader_cases']}",
                "Citation grounded": f"{row['reader_grounded']}/{row['reader_cases']}",
            }
        )
    for name, rows in [("table1", table1), ("table2", table2)]:
        csv_table(args.out / (name + ".csv"), rows)
        latex_table(
            args.out / (name + ".tex"),
            list(rows[0]),
            [list(row.values()) for row in rows],
        )
    csv_table(
        args.out / "paired_tests.csv",
        [{"test": key, **value} for key, value in final["primary_tests"].items()],
    )
    csv_table(args.out / "long_context.csv", final["long_context"])
    csv_table(
        args.out / "collisions.csv",
        [{"method": key, **value} for key, value in final["collisions"].items()],
    )
    qualification = [
        {"model": model, **result["qualification"]}
        for model, result in final["readers"].items()
    ]
    csv_table(args.out / "qualification.csv", qualification)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "figure.py"),
            "--analysis",
            str(args.out / "final/analysis.json"),
            "--out",
            str(args.out),
        ],
        check=True,
    )
    print(args.out)


if __name__ == "__main__":
    main()
