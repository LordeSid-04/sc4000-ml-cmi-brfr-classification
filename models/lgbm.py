"""LightGBM on the 795 summary features. Defaults = the team's reference settings from preprocessing validation;
tuned settings (models.train.tune) arrive as keyword overrides."""


def make(seed=0, trees=500, **kw):
    import lightgbm as lgb
    p = dict(learning_rate=0.05, n_estimators=trees, num_leaves=31, min_child_samples=20, colsample_bytree=0.3,
             subsample=0.8, subsample_freq=1, reg_lambda=1.0)
    return lgb.LGBMClassifier(**{**p, **kw}, random_state=seed, verbose=-1)
