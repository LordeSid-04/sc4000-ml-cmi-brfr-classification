"""LightGBM on the 795 summary features (reference settings from data/processed/validation.json)."""


def make(seed=0, trees=500):
    import lightgbm as lgb
    return lgb.LGBMClassifier(learning_rate=0.05, n_estimators=trees, num_leaves=31, min_child_samples=20,
                              colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
