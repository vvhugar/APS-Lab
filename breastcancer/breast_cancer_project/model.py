"""
Model creation and training.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_model():
    """Create the StandardScaler + LogisticRegression pipeline."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )


def train_model(model, x_train, y_train):
    """Fit the model on the training data."""
    model.fit(x_train, y_train)
    return model
