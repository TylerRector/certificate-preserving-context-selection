import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from storage import ROOT, save


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(name, directory):
    models = json.loads((ROOT / "data/models.json").read_text())
    lock = models[name]
    path = directory / name
    snapshot_download(
        repo_id=lock["repo"],
        revision=lock["revision"],
        local_dir=path,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
    )
    for filename, expected in lock["files"].items():
        assert checksum(path / filename) == expected, filename
    save(path / "model_lock.json", lock)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        choices=["qwen3-14b", "qwen3-32b", "recomp-nq", "recomp-hotpotqa"],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    for name in args.model:
        download(name, args.out)


if __name__ == "__main__":
    main()
