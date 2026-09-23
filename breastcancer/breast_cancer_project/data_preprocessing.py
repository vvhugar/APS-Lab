"""
Data loading and preprocessing for the breast cancer classification project.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    """Load the sklearn breast cancer dataset and create X and y."""
    data = load_breast_cancer()

    x = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Preserve the notebook's target encoding:
    # original sklearn target 0 (malignant) -> 1
    # original sklearn target 1 (benign) -> 0
    y = pd.Series(
        (data.target == 0).astype(int),
        name="malignant"
    )

    return data, x, y


def get_class_distribution(y, data):
    """Create the class-count and probability table used in the notebook."""
    class_counts = y.value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Class": data.target_names,
        "Count": class_counts,
        "Probability": class_counts.values / len(y)
    })

    return class_distribution


def split_data(x, y, test_size=0.2, random_state=42):
    """Split the data using the notebook's stratified train/test split."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def print_split_information(y_train, y_test):
    """Print training/testing sizes and class proportions."""
    print("Training size: ", len(y_train))
    print("Testing size: ", len(y_test))

    print("\nTraining proportions:")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nTesting proportions:")
    print(y_test.value_counts(normalize=True).sort_index())
