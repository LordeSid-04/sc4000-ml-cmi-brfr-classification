"""Random forest on the summary features (bagging baseline; NaN -> -999)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def make(seed=0, trees=500, **kw):
    p = dict(n_estimators=trees, max_features='sqrt', min_samples_leaf=2)
    return make_pipeline(SimpleImputer(strategy='constant', fill_value=-999),
                         RandomForestClassifier(**{**p, **kw}, n_jobs=-1, random_state=seed))
