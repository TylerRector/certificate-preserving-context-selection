import argparse
import subprocess
import sys
from pathlib import Path

from storage import ROOT


def execute(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["final", "recomp"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    parser.add_argument("--per-seed", type=int, default=512)
    args = parser.parse_args()
    models = (
        ["qwen3-14b", "qwen3-32b"]
        if args.experiment == "final"
        else ["recomp-nq", "recomp-hotpotqa", "qwen3-32b"]
    )
    if args.download:
        from download_models import download

        for model in models:
            download(model, args.models)
    output = args.out / args.experiment
    if args.experiment == "final":
        execute("generate.py", "--out", output / "cpu", "--per-seed", args.per_seed)
        for size in ["14b", "32b"]:
            execute(
                "reader.py",
                "--model",
                args.models / ("qwen3-" + size),
                "--data",
                output / "cpu",
                "--out",
                output / size,
            )
    else:
        execute("prepare_recomp.py", "--out", output / "input")
        execute(
            "recomp.py",
            "--input",
            output / "input",
            "--out",
            output / "compression",
            "--models-root",
            args.models,
        )
        execute(
            "reader.py",
            "--model",
            args.models / "qwen3-32b",
            "--data",
            output / "compression",
            "--out",
            output / "32b",
        )


if __name__ == "__main__":
    main()
