"""CMI sensor preprocessing: cleaning, canonical frames, features, labels, shaping."""
from .step1_cleaning import CleanSequence, DataError, clean_sequence
from .pipeline import PipelineConfig, Processed, baseline_features, preprocess

__all__ = ['CleanSequence', 'DataError', 'clean_sequence', 'PipelineConfig',
           'Processed', 'baseline_features', 'preprocess']
