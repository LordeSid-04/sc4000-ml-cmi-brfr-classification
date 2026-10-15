"""The ablation study (guide section 9): which pipeline each row uses.

Rows A-G add one preprocessing step at a time. The "G-..." rows remove a single
step from the full pipeline (leave-one-out) or swap one design choice.
"""
from dataclasses import replace

import numpy as np

from .augmentation import drop_sensors
from .step1_cleaning import DataError, clean_sequence
from .constants import THM, TOF
from .step3_features import FeatureConfig
from .step2_orientation import canonicalize
from .pipeline import PipelineConfig, baseline_features, featurize
from .step5_shaping import tree_summary

_PLAIN = FeatureConfig(gravity=False, engineered=False)
_B = PipelineConfig(fix_upside_down=False, mirror_left=False, features=_PLAIN)
# Rows A-G and the leave-one-out variants were measured with smoothing on (sigma=1),
# the guide's original plan. Row H is the final pipeline (PipelineConfig() defaults).
_SMOOTH = 1.0
FULL = PipelineConfig(features=FeatureConfig(smooth_sigma=_SMOOTH))
FINAL = PipelineConfig()

# name -> (description, PipelineConfig or None for raw baseline, use_tail)
ROWS = {
    'A': ('Baseline: raw columns, NaN -> 0', None, False),
    'B': ('+ cleaning: order, gap filling, quaternion continuity, ToF proximity, THM/ToF summaries', _B, False),
    'C': ('+ keep-end statistics (last 25% of the recording)', _B, True),
    'D': ('+ upside-down subjects corrected', replace(_B, fix_upside_down=True), True),
    'E': ('+ left-handed subjects mirrored (all sensors)', replace(_B, fix_upside_down=True, mirror_left=True), True),
    'F': ('+ gravity removal with low-pass fallback', replace(FULL, features=FeatureConfig(engineered=False, smooth_sigma=_SMOOTH)), True),
    'G': ('+ smoothing and engineered features', FULL, True),
    'H': ('final pipeline: G without smoothing', FINAL, True),
}
VARIANTS = {
    'G-no_continuity': ('G without quaternion sign continuity', replace(FULL, continuity=False), True),
    'G-tof_near': ('G with ToF -1 treated as touching (previous cleaning)', replace(FULL, tof_no_object='near'), True),
    'G-no_upside_down': ('G without the upside-down fix', replace(FULL, fix_upside_down=False), True),
    'G-no_mirror': ('G without left-hand mirroring', replace(FULL, mirror_left=False), True),
    'G-no_gravity': ('G without gravity removal', replace(FULL, features=FeatureConfig(gravity=False, smooth_sigma=_SMOOTH)), True),
    'G-no_regions': ('G without ToF 3x3 regions', replace(FULL, features=FeatureConfig(tof_regions=False, smooth_sigma=_SMOOTH)), True),
}
ALL = {**ROWS, **VARIANTS}


def _clean(rows, config):
    try:
        return clean_sequence(rows, strict=True, continuity=config.continuity, tof_no_object=config.tof_no_object)
    except DataError:
        return clean_sequence(rows, strict=False, continuity=config.continuity, tof_no_object=config.tof_no_object)


def tree_rows_for_sequence(rows, handedness, names):
    """Tree features (all sensors and IMU-only) of one recording for each named experiment."""
    sid = rows['sequence_id'].iloc[0]
    subject = rows['subject'].iloc[0]
    cache = {}
    result = {}
    for name in names:
        _, config, use_tail = ALL[name]
        if config is None:
            raw = baseline_features(rows)
            imu = raw.copy()
            imu[THM + TOF] = 0.0
            result[name] = ({'sequence_id': sid, **tree_summary(raw, use_tail=use_tail)},
                            {'sequence_id': sid, **tree_summary(imu, use_tail=use_tail)})
            continue
        key = (config.continuity, config.tof_no_object)
        if key not in cache:
            cache[key] = _clean(rows, config)
        seq = canonicalize(cache[key], subject=subject,
                           handedness=handedness if config.mirror_left else None,
                           upside_down=None if config.fix_upside_down else False)
        full = featurize(seq, config)[0]
        imu = featurize(drop_sensors(seq, True, False), config)[0]
        result[name] = ({'sequence_id': sid, **tree_summary(full, use_tail=use_tail)},
                        {'sequence_id': sid, **tree_summary(imu, use_tail=use_tail)})
    return result


def lgbm_params(seed=42, n_classes=18):
    return dict(objective='multiclass', num_class=n_classes, learning_rate=0.05,
                num_leaves=31, min_child_samples=20, feature_fraction=0.3,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                n_estimators=500, random_state=seed, verbose=-1, n_jobs=6)


def cross_validate_lgbm(x_full, x_imu, y, folds, seed=42, exclude_train=None):
    """Train on all-sensor + IMU-only copies of the training folds; score both
    validation versions. Returns per-fold scores and out-of-fold probabilities."""
    import lightgbm as lgb
    from .constants import GESTURES
    from .step4_labels import competition_score
    n_classes = len(GESTURES)
    oof_full = np.zeros((len(y), n_classes))
    oof_imu = np.zeros((len(y), n_classes))
    scores = []
    for k in sorted(np.unique(folds)):
        tr = folds != k
        if exclude_train is not None:
            tr &= ~exclude_train
        va = folds == k
        model = lgb.LGBMClassifier(**lgbm_params(seed, n_classes))
        model.fit(np.vstack([x_full[tr], x_imu[tr]]), np.r_[y[tr], y[tr]])
        oof_full[va] = model.predict_proba(x_full[va])
        oof_imu[va] = model.predict_proba(x_imu[va])
        true = [GESTURES[i] for i in y[va]]
        s_full = competition_score(true, [GESTURES[i] for i in oof_full[va].argmax(1)])
        s_imu = competition_score(true, [GESTURES[i] for i in oof_imu[va].argmax(1)])
        scores.append({'fold': int(k), 'full': s_full[0], 'full_binary': s_full[1], 'full_macro': s_full[2],
                       'imu': s_imu[0], 'imu_binary': s_imu[1], 'imu_macro': s_imu[2]})
        print(f'    fold {k}: full {s_full[0]:.4f}  imu {s_imu[0]:.4f}', flush=True)
    return scores, oof_full, oof_imu
