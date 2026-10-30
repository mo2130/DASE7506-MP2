# DASE7506 MP2 — Selecting Reliable LLM Reasoning

Student ID: **[3036706741]**

Final test accuracy: **83.2976** (389 / 467)

## Method

For each question, four candidate traces are represented by fixed 1024-dimensional
Qwen3-0.6B embeddings. We

1. subtract the within-question mean of the four candidates (contrast features),
2. score every candidate with a ridge probe trained on the training split,
3. mask candidates whose parsed final answer is missing, non-integer, or zero,
4. select the highest-scoring remaining candidate.

The probe is trained only on candidates whose final answer is a non-zero integer
(17,722 of 19,020 training candidates). In the supplied splits no correct final
answer is fractional or zero; negative integers are kept eligible.

## Repository layout

```
code/
  common.py                 fixed parser / loader
  predict.py                prediction runner
  score_predictions.py      independent scorer
  check_data.py             data integrity check
  submission.py             final method (loads the checkpoint)
  train_probe.py            training script (ridge probe)
  majority.py               majority-vote baseline
  runs/model_bundle/probe.npz   submitted checkpoint (12,279 bytes)
  runs/test-score.json      official test score
  runs/test-predictions.json
report/
  MP2_Report_EN.pdf         report (7 pages)
README.md
```

The course-provided `data/` directory is **not** duplicated here (about 100 MB).
Copy it from the official course package into `code/data/` before running.

## Environment

- Python 3.12
- NumPy 2.5.3 (`pip install -r requirements.txt`)
- CPU only, four threads

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python check_data.py               # expect "status": "passed"
```

## Reproduce

From `code/`:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python train_probe.py
python predict.py --method submission.py --checkpoint runs/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

On Windows cmd:

```bat
set OMP_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set MKL_NUM_THREADS=4
python train_probe.py
python predict.py --method submission.py --checkpoint runs\model_bundle --split test --output runs\test-predictions.json
python score_predictions.py --split test --predictions runs\test-predictions.json --output runs\test-score.json
```

Expected:

```json
{"correct": 745, "n": 798, "accuracy": 93.35839598997494}
{"correct": 389, "n": 467, "accuracy": 83.29764453961457}
```

Training is deterministic (closed-form ridge, no random seed). The byte hash of
`probe.npz` may change across re-saves because of ZIP metadata, but the weights
are equivalent.

## Submitted checkpoint

- File: `code/runs/model_bundle/probe.npz`
- Size: 12,279 bytes
- SHA-256: `1c1f59dbb9bfca89ef327bf42629d7b2e575a08a601fb8dfbf888e03b39fa19f`
- Contents: `w` (1024 float32), `mu`, `sd` (1024 each), intercept `b`, feature mode `contrast`

Prediction takes about 0.1–0.2 s for the 467-question test bank on four CPU
threads, well inside the 30 s limit; the asset is well inside 16 MiB.

## AI disclosure

AI assistance was used for code implementation, analysis scripts.
## Report

`report/3036706741-Zimo Zhang-DASE7506 MP2
