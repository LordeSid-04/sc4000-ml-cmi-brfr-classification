"""Per-person gesture balancing at test time (post-processing, nothing is trained).

Every participant follows the same protocol: in training, each person performs each of the 8 BFRBs 8 times and each
other gesture 2 or 6 times. The test server sends a person's recordings one at a time, so after each prediction we
count what this person has been assigned so far and lower the probability of gestures already used up:

    p'(g) = p(g) * max(quota(g) - count(g) + a, 0.05) / (quota(g) + a)

a = strength (0 = off). A large a changes little; a small a enforces the quota hard. It uses only the person's
*previous predictions* (never labels), so it is causal and valid on Kaggle. Idea from the 7th-place talk
(per-person balancing); the rule and its nested evaluation are ours."""
import numpy as np

STRENGTHS = (0, 1, 2, 4, 8, 16)


def quota(seqs):
    """Typical number of recordings of each gesture per person (median over training people), length 18."""
    c = seqs.groupby(['subject', 'gesture_id']).size().unstack(fill_value=0).reindex(columns=range(18), fill_value=0)
    return c.median().to_numpy(float)


def adjust(p, counts, q, a):
    return p if not a else p * np.maximum(q - counts + a, 0.05) / (q + a)


def simulate(pf, pi, subject, q, a, seed):
    """Replay the test server on out-of-fold predictions: each person's recordings arrive in a random order, and each
    recording is randomly all-sensor or motion-only (about half the test set is motion-only). -> predicted gesture ids."""
    rng = np.random.default_rng(seed)
    p = np.where(rng.random(len(pf))[:, None] < 0.5, pi, pf)
    pred = np.zeros(len(p), int)
    for s in np.unique(subject):
        idx = rng.permutation(np.flatnonzero(subject == s))
        counts = np.zeros(18)
        for i in idx:
            pred[i] = g = int(adjust(p[i], counts, q, a).argmax())
            counts[g] += 1
    return pred
