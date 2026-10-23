"""XGBoost on the summary features (second boosting implementation; GPU when available)."""
import torch


def make(seed=0, trees=500, **kw):
    import xgboost as xgb
    p = dict(n_estimators=trees, learning_rate=0.05, max_depth=6, colsample_bytree=0.3, subsample=0.8,
             min_child_weight=2, reg_lambda=1.0)
    return xgb.XGBClassifier(**{**p, **kw}, tree_method='hist', random_state=seed,
                             device='cuda' if torch.cuda.is_available() else 'cpu')
