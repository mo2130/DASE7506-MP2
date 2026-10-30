"""Train a linear probe on integer-answer candidates only (v3)."""
from fractions import Fraction
from pathlib import Path

import numpy as np

import common

DATA = Path(__file__).resolve().parent / 'data'
MODE = 'contrast'     # 'raw' or 'contrast'
LAM = 1000.0


def load_train():
    bank, arrays = common.load_inputs('train', DATA)
    gold = common.read_jsonl(DATA / 'gsm8k_train.jsonl')
    answers = [[common.parse_answer(c['text']) for c in row['candidates']]
               for row in bank]
    golds = [common.parse_answer(row['answer']) for row in gold]
    labels = np.array([[a is not None and a == g for a in row]
                       for row, g in zip(answers, golds)], dtype=np.float32)
    keep = np.array([[a is not None and Fraction(a).denominator == 1 and Fraction(a) != 0
                      for a in row] for row in answers])
    emb = arrays['embeddings'].astype(np.float32)
    return emb, labels, keep


def make_features(emb, mode):
    n = emb.shape[0]
    x = emb.reshape(n * 4, -1)
    if mode == 'raw':
        return x
    center = emb.mean(axis=1, keepdims=True).repeat(4, axis=1).reshape(n * 4, -1)
    return x - center


def main():
    emb, labels, keep = load_train()
    x = make_features(emb, MODE)
    mask = keep.reshape(-1)
    x, y = x[mask], labels.reshape(-1)[mask].astype(np.float64)
    print('train embeddings:', emb.shape, '| kept candidates:', int(mask.sum()),
          '| positive rate:', round(float(y.mean()), 3))

    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + LAM * np.eye(z.shape[1]),
                        z.T @ (y - y.mean())).astype(np.float32)

    out = Path('runs/model_bundle')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'probe.npz', w=w, mu=mu.astype(np.float32),
                        sd=sd.astype(np.float32), b=np.float32(y.mean()),
                        mode=np.array(MODE))
    print('mode:', MODE, '| features:', x.shape[1], '| saved:', out / 'probe.npz')


if __name__ == '__main__':
    main()
