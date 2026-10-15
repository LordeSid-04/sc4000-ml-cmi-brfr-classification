"""Step 9 of the guide: every label type, built from train-only columns.

These are training TARGETS only. None of them may be used as a model input,
because the hidden test set does not contain the columns they come from.
"""
import numpy as np
import pandas as pd

from .constants import GESTURES, SCORE_CLASSES, TARGET_GESTURES

GESTURE_INDEX = {g: i for i, g in enumerate(GESTURES)}
SCORE_INDEX = {g: i for i, g in enumerate(SCORE_CLASSES)}
PHASES = ['transition', 'pause', 'gesture']


def score_class(gesture):
    """9-class label used by the competition's macro F1."""
    return gesture if gesture in TARGET_GESTURES else 'non_target'


def phase_of(behavior):
    """Map the per-row `behavior` text to transition / pause / gesture."""
    b = str(behavior).lower()
    if 'perform' in b:
        return 'gesture'
    if 'at target' in b:
        return 'pause'
    return 'transition'


def initial_state(behaviors):
    """How the recording starts: 'relaxed' if the first behaviour mentions relaxing
    (hand starts at rest), otherwise 'ready'."""
    first = str(behaviors.iloc[0]).lower() if len(behaviors) else ''
    return 'relaxed' if 'relax' in first else 'ready'


def sequence_labels(meta):
    """Labels for one recording from its (train) meta rows."""
    gesture = meta['gesture'].iloc[0]
    out = {
        'gesture': gesture,
        'gesture_id': GESTURE_INDEX[gesture],
        'is_target': int(gesture in TARGET_GESTURES),
        'score_class': score_class(gesture),
        'score_id': SCORE_INDEX[score_class(gesture)],
    }
    if 'orientation' in meta:
        out['orientation'] = meta['orientation'].iloc[0]
    if 'behavior' in meta:
        out['initial_state'] = initial_state(meta['behavior'])
        if 'orientation' in meta:
            out['combined'] = f"{gesture} | {out['orientation']} | {out['initial_state']}"
    return out


def row_phase(meta):
    """Per-row phase index (0 transition, 1 pause, 2 gesture) or None if unavailable."""
    if 'behavior' in meta:
        return np.array([PHASES.index(phase_of(b)) for b in meta['behavior']], dtype=np.int8)
    if 'phase' in meta:
        return np.where(meta['phase'].astype(str).str.lower() == 'gesture', 2, 0).astype(np.int8)
    return None


def competition_score(true_gestures, predicted_gestures):
    """0.5 * binary F1 (target vs not) + 0.5 * macro F1 over the 9 score classes."""
    from sklearn.metrics import f1_score
    true = pd.Series(true_gestures).map(score_class)
    pred = pd.Series(predicted_gestures).map(score_class)
    binary = f1_score(true != 'non_target', pred != 'non_target')
    macro = f1_score(true, pred, average='macro', zero_division=0)
    return 0.5 * binary + 0.5 * macro, binary, macro
