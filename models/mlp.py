"""MLP on the summary features: same inputs as the trees, so it isolates the learner from the representation."""
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make(seed=0, trees=None):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         MLPClassifier((512, 256), alpha=1e-3, early_stopping=True, max_iter=200, random_state=seed))
