# Certificate-Preserving Context Selection

Code and saved data for **Certificate-Preserving Context Selection for Multi-Agent Oversight**. Includes the four-seed experiment, long-context checks, 14B/32B reader outputs, and preliminary RECOMP tests. Methods use the paper's names: **Multi-candidate certificate selection**, **Certificate-greedy selection**, and **Okapi BM25**.

Recreate Figure 1, Tables 1–2, paired tests, and data audits with Python 3.11+:

```bash
pip install -r requirements.txt
python reproduce.py
python -m unittest discover -s tests
```

Results are written to `outputs/`. Raw records and generations are in `data/`; large logs use gzip. Both certificate selectors recover 1,966/2,048 primary verdicts. Both readers fail qualification; the eight-case RECOMP comparison is exploratory.

To rerun inference with the recorded model revisions, install `requirements-inference.txt` and use a suitable GPU:

```bash
python run.py final --download
python run.py recomp --download
python reproduce.py --final runs/final --recomp runs/recomp
```

Fresh runs go to `runs/`; downloaded weights go to `models/`. The draft paper, model weights, and machine-specific deployment files are not included. RECOMP attribution is in `NOTICE` and `licenses/RECOMP`.
