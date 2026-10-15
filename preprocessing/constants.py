"""Column names, sensor groups and label vocabularies shared by the whole pipeline."""

ACC = ['acc_x', 'acc_y', 'acc_z']
ROT = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
THM = [f'thm_{i}' for i in range(1, 6)]
TOF = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
SENSORS = ACC + ROT + THM + TOF
ID_COLUMNS = ['sequence_id', 'sequence_counter']

# Columns that exist only in train.csv. They may be used as training targets,
# never as model inputs, because the hidden test set does not have them.
TRAIN_ONLY = ['gesture', 'sequence_type', 'orientation', 'behavior', 'phase']

TOF_MAX = 254.0          # largest valid ToF reading; -1 means "no object in range"
GRAVITY = 9.81           # m/s^2
SAMPLE_RATE_HZ = 10.0    # approximate; only used to express filters in seconds

TARGET_GESTURES = [
    'Above ear - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Neck - pinch skin',
    'Neck - scratch',
    'Cheek - pinch skin',
]
NON_TARGET_GESTURES = [
    'Drink from bottle/cup',
    'Glasses on/off',
    'Pull air toward your face',
    'Pinch knee/leg skin',
    'Scratch knee/leg skin',
    'Write name on leg',
    'Text on phone',
    'Feel around in tray and pull out an object',
    'Write name in air',
    'Wave hello',
]
GESTURES = TARGET_GESTURES + NON_TARGET_GESTURES
# The competition's macro F1 collapses all non-target gestures into one class.
SCORE_CLASSES = TARGET_GESTURES + ['non_target']
