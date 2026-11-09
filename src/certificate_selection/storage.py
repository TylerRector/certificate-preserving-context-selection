import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = ROOT / "assets/tokenizer"


def source_hashes():
    package = Path(__file__).resolve().parent
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }


def resolve(path):
    path = Path(path)
    return path if path.exists() else Path(str(path) + ".gz")


def open_text(path):
    path = resolve(path)
    return (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8")
    )


def iter_jsonl(path):
    with open_text(path) as file:
        for line in file:
            yield json.loads(line)


def load_jsonl(path):
    return list(iter_jsonl(path)) if resolve(path).exists() else []


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def append(path, value):
    with Path(path).open("a") as file:
        file.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def copy_records(source, destination):
    with open_text(source) as inp, Path(destination).open("w") as out:
        for line in inp:
            out.write(line)
