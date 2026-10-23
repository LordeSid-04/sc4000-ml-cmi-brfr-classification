"""Ways to combine a network with a tree model. All three are fitted on OTHER folds' out-of-fold predictions only,
then applied to the held-out fold, so their reported scores stay honest.
  h1  weighted probability average (the one deployed on Kaggle: one weight per sensor setting)
  h2  logistic-regression stacker on both models' log-probabilities
  h3  LightGBM on the summary features + the network's probabilities"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.common import score

GRID = np.linspace(0, 1, 21)


def best_weight(y, pa, pb):
    return float(GRID[np.argmax([score(y, w * pa + (1 - w) * pb) for w in GRID])])


def fit_predict(y, fold, pa, pb, X):
    """pa = network probs, pb = tree probs, X = summary features of this sensor setting. -> {h: (N, 18) probs}"""
    import lightgbm as lgb
    out = {h: np.zeros_like(pa) for h in ('h1', 'h2', 'h3')}
    logp = np.log(np.c_[pa, pb] + 1e-6)
    F = pd.concat([X.reset_index(drop=True), pd.DataFrame(pa, columns=[f'nn{i}' for i in range(18)])], axis=1)
    for k in range(5):
        tr, va = fold != k, fold == k
        w = best_weight(y[tr], pa[tr], pb[tr])
        out['h1'][va] = w * pa[va] + (1 - w) * pb[va]
        out['h2'][va] = LogisticRegression(C=1.0, max_iter=500).fit(logp[tr], y[tr]).predict_proba(logp[va])
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                               subsample=0.8, subsample_freq=1, verbose=-1).fit(F[tr], y[tr])
        out['h3'][va] = m.predict_proba(F[va])
    return out
