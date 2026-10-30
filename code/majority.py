from collections import Counter

import common


def predict(bank, arrays, checkpoint):
    picks = []
    for row in bank:
        answers = [common.parse_answer(c['text']) for c in row['candidates']]
        votes = Counter(a for a in answers if a is not None)
        if not votes:
            picks.append(0)
            continue
        top = max(votes.values())
        pick = next(i for i, a in enumerate(answers)
                    if a is not None and votes[a] == top)
        picks.append(pick)
    return picks