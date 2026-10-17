"""Step 13 of the guide: one preprocess() for training and for Kaggle inference.

    out = preprocess(sequence_df, demographics_df)
    out.features     # (n, F) per-row features, NaN where a sensor is unavailable
    out.tof_images   # (n, 5, 8, 8) proximity images
    out.clean        # the canonical CleanSequence (for augmentation)

It handles all four sensor situations (full, no thermopile/ToF, no rotation,
neither) and never needs labels or train-only columns.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .step1_cleaning import clean_sequence
from .step3_features import FeatureConfig, sequence_features, tof_images
from .step2_orientation import canonicalize


@dataclass(frozen=True)
class PipelineConfig:
    continuity: bool = True
    tof_no_object: str = 'far'
    # The proposed corrections were selected using labels from all subjects.
    # Keep them opt-in for exploratory work; held-out evaluation uses neither.
    fix_upside_down: bool = False
    mirror_left: bool = False
    features: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class Processed:
    sequence_id: str
    features: pd.DataFrame
    tof_images: np.ndarray
    clean: object


def to_pandas(frame):
    """Kaggle passes polars DataFrames; everything here uses pandas."""
    return frame.to_pandas() if hasattr(frame, 'to_pandas') else frame


def demographic_values(demographics, subject=None):
    """(subject, handedness) from a demographics table (one or many rows)."""
    if demographics is None:
        return subject, None
    demo = to_pandas(demographics)
    if subject is not None and 'subject' in demo:
        demo = demo[demo['subject'] == subject]
    elif len(demo) != 1:
        raise ValueError('A subject column or exactly one demographics row is required')
    if len(demo) == 0:
        return subject, None
    row = demo.iloc[0]
    handedness = row.get('handedness')
    handedness = None if handedness is None or pd.isna(handedness) else int(handedness)
    return row.get('subject', subject), handedness


def featurize(seq, config=PipelineConfig()):
    return sequence_features(seq, config.features), tof_images(seq)


def canonical(sequence, demographics=None, config=PipelineConfig(), strict=False):
    sequence = to_pandas(sequence)
    seq = clean_sequence(sequence, strict=strict, continuity=config.continuity,
                         tof_no_object=config.tof_no_object)
    subject = sequence['subject'].iloc[0] if 'subject' in sequence else None
    subject, handedness = demographic_values(demographics, subject)
    return canonicalize(
        seq, subject=subject,
        handedness=handedness if config.mirror_left else None,
        upside_down=None if config.fix_upside_down else False)


def preprocess(sequence, demographics=None, config=PipelineConfig(), strict=False):
    seq = canonical(sequence, demographics, config, strict)
    features, images = featurize(seq, config)
    return Processed(seq.sequence_id, features, images, seq)

