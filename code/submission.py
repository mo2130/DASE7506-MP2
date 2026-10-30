"""v3 submission: probe trained on integer answers; mask non-integer answers."""
from fractions import Fraction
from pathlib import Path

import numpy as np

import common


def make_features(emb, mode):
    n = emb.shape[0]
    x = emb.reshape(n * 4, -1)
    if mode == 'raw':
        return x
    center = emb.mean(axis=1, keepdims=True).repeat(4, axis=1).reshape(n * 4, -1)
    return x - center


def predict(bank, arrays, checkpoint):
    bundle = np.load(Path(checkpoint) / 'probe.npz')
    w, mu, sd = bundle['w'], bundle['mu'], bundle['sd']
    b = float(bundle['b'])
    mode = str(bundle['mode'])
    emb = arrays['embeddings'].astype(np.float32)
    x = make_features(emb, mode)
    scores = (((x - mu) / sd) @ w + b).reshape(len(bank), 4)

    all_answers = [[common.parse_answer(c['text']) for c in row['candidates']]
                   for row in bank]
    usable = np.array([[a is not None and Fraction(a).denominator == 1 and Fraction(a) != 0
                        for a in row] for row in all_answers])
    scores[~usable] = -np.inf
    return [int(i) for i in scores.argmax(axis=1)]
