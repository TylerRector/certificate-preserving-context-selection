import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from storage import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis", type=Path, default=ROOT / "outputs/final/analysis.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    methods = ["multi_candidate", "certificate_greedy", "okapi_bm25"]
    rates = [
        100 * data["cpu"]["all"][method]["certified"] / data["cpu"]["all"][method]["n"]
        for method in methods
    ]
    axes[0].bar(
        range(3), rates, color=["black", "0.55", "white"], edgecolor="black", width=0.58
    )
    for index, value in enumerate(rates):
        axes[0].text(index, value + 3, f"{value:.1f}", ha="center")
    axes[0].set_xticks(range(3), ["Multi-cand.", "Cert.-greedy", "Okapi BM25"])
    axes[0].set_ylabel("Certified decisions (%)")
    axes[0].set_title("Evidence recovered")
    points = [
        ("14b", "multi_candidate"),
        ("14b", "certificate_greedy"),
        ("32b", "multi_candidate"),
        ("32b", "certificate_greedy"),
    ]
    raw, grounded = [], []
    for model, method in points:
        row = data["readers"][model]["metrics"]["all"][method]
        raw.append(100 * row["correct"] / row["n"])
        grounded.append(100 * row["citation_grounded_correct"] / row["n"])
    x = np.arange(len(points))
    axes[1].scatter(x, raw, color="black", marker="o", label="Correct verdict")
    axes[1].scatter(
        x,
        grounded,
        facecolors="none",
        edgecolors="black",
        marker="s",
        label="Correct + cited proof",
    )
    axes[1].set_xticks(x, ["14B\nMulti", "14B\nGreedy", "32B\nMulti", "32B\nGreedy"])
    axes[1].set_title("Reader accuracy and cited proof")
    axes[1].set_ylabel("Correct decisions (%)")
    axes[1].legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.set_ylim(0, 112)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.spines[["top", "right"]].set_visible(False)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "figure1.png", dpi=300)
    fig.savefig(args.out / "figure1.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
