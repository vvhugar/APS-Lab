"""
Prediction and evaluation functions.

This module combines the notebook's prediction, probability,
threshold analysis, confusion-matrix, and metric calculations.
"""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def get_probabilities(model, x_test):
    """Generate class probabilities for the test set."""
    return model.predict_proba(x_test)


def create_prediction_results(probabilities, y_test):
    """Create the prediction table used in the notebook."""
    results = pd.DataFrame({
        "Actual_class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1],
    })

    results["Actual label"] = results["Actual_class"].map({
        0: "Malignant",
        1: "Benign",
    })

    return results


def add_threshold_predictions(results, threshold=0.5):
    """Add malignant predictions and readable predicted labels."""
    results = results.copy()

    results["Predicted malignant"] = (
        results["P_malignant"] >= threshold
    ).astype(int)

    results["Predicted_label"] = results["Predicted malignant"].map({
        1: "malignant",
        0: "benign",
    })

    return results


def count_malignant_predictions(results, thresholds=(0.30, 0.50, 0.70)):
    """Print the number of predicted malignant cases at each threshold."""
    counts = {}

    for threshold in thresholds:
        predictions = (
            results["P_malignant"] >= threshold
        ).astype(int)

        counts[threshold] = int(predictions.sum())

        print(
            f"Threshold = {threshold}: "
            f"Predicted malignant cases = {predictions.sum()}"
        )

    return counts


def evaluate_thresholds(
    probabilities,
    y_test,
    thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)
):
    """
    Calculate TN, FP, FN, TP, accuracy, precision, recall,
    and F1-score for each probability threshold.
    """
    # The notebook evaluates malignant as the positive class.
    actual_malignant = (y_test.values == 0).astype(int)

    metrics_data = []

    for threshold in thresholds:
        predicted_malignant = (
            probabilities[:, 0] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            actual_malignant.tolist(),
            predicted_malignant.tolist()
        )

        tn, fp, fn, tp = (
            cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        )

        accuracy = accuracy_score(
            actual_malignant,
            predicted_malignant
        )
        precision = precision_score(
            actual_malignant,
            predicted_malignant,
            zero_division=0
        )
        recall = recall_score(
            actual_malignant,
            predicted_malignant,
            zero_division=0
        )
        f1 = f1_score(
            actual_malignant,
            predicted_malignant,
            zero_division=0
        )

        metrics_data.append({
            "Threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        })

    return pd.DataFrame(metrics_data)


def print_confusion_matrices(
    probabilities,
    y_test,
    thresholds=(0.30, 0.50, 0.70)
):
    """Print confusion matrices for selected thresholds."""
    actual_malignant = (y_test.values == 0).astype(int)

    matrices = {}

    for threshold in thresholds:
        predicted_malignant = (
            probabilities[:, 0] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            actual_malignant.tolist(),
            predicted_malignant.tolist()
        )

        matrices[threshold] = cm

        print(f"\nConfusion Matrix for Threshold = {threshold}:")
        print(cm)

    return matrices
