"""Loading the competition files from the Kaggle ZIP or an extracted folder."""
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_LOCATIONS = [
    Path('data/raw/cmi-detect-behavior-with-sensor-data.zip'),
    Path('data/raw'),
    Path('/kaggle/input/cmi-detect-behavior-with-sensor-data'),
]


def find_data(path=None):
    candidates = [Path(path)] if path else DEFAULT_LOCATIONS
    for p in candidates:
        if p.is_file() and p.suffix == '.zip':
            return p
        if p.is_dir() and (p / 'train.csv').exists():
            return p
        if p.is_dir():
            zips = sorted(p.glob('*.zip'))
            if zips:
                return zips[0]
    raise FileNotFoundError(
        'Competition data not found. Put cmi-detect-behavior-with-sensor-data.zip '
        '(or the extracted train.csv) in data/raw/, or pass --data.')


def read_table(source, name):
    source = Path(source)
    if source.suffix == '.zip':
        with zipfile.ZipFile(source) as z:
            match = [n for n in z.namelist() if n.endswith(name)]
            if not match:
                raise FileNotFoundError(f'{name} not in {source}')
            with z.open(sorted(match, key=len)[0]) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(source / name)


def load_train(path=None):
    source = find_data(path)
    train = read_table(source, 'train.csv')
    demographics = read_table(source, 'train_demographics.csv')
    return train, demographics


def load_test(path=None):
    source = find_data(path)
    return read_table(source, 'test.csv'), read_table(source, 'test_demographics.csv')


def iter_sequences(train):
    """Yield (sequence_id, rows) in a stable order."""
    for sid, rows in train.groupby('sequence_id', sort=True):
        yield sid, rows
