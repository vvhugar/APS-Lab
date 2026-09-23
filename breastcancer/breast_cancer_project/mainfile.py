
from data_preprocessing import (
    load_and_prepare_data,
    get_class_distribution,
    split_data,
    print_split_information,
)

from model import build_model, train_model

from prediction_evaluation import (
    get_probabilities,
    create_prediction_results,
    add_threshold_predictions,
    count_malignant_predictions,
    print_confusion_matrices,
    evaluate_thresholds,
)

from visualization import plot_class_distribution


def main():
    # ---------------------------------------------------------
    # 1. Load and prepare data
    # ---------------------------------------------------------
    data, x, y = load_and_prepare_data()

    print(y.value_counts())
    print("feature matrix shape: ", x.shape)
    print("target shape: ", y.shape)
    print("class names: ", data.target_names)

    # ---------------------------------------------------------
    # 2. Class distribution
    # ---------------------------------------------------------
    class_distribution = get_class_distribution(y, data)
    print(class_distribution)

    plot_class_distribution(class_distribution)

    # ---------------------------------------------------------
    # 3. Train/test split
    # ---------------------------------------------------------
    x_train, x_test, y_train, y_test = split_data(x, y)

    print_split_information(y_train, y_test)

    # ---------------------------------------------------------
    # 4. Build and train model
    # ---------------------------------------------------------
    model = build_model()
    model = train_model(model, x_train, y_train)

    print("\nModel:")
    print(model)

    # ---------------------------------------------------------
    # 5. Generate probabilities
    # ---------------------------------------------------------
    probabilities = get_probabilities(model, x_test)

    print("\nFirst 5 probability predictions:")
    print(probabilities[:5])

    # ---------------------------------------------------------
    # 6. Create prediction results
    # ---------------------------------------------------------
    results = create_prediction_results(probabilities, y_test)

    print("\nPrediction results:")
    print(
        results[
            ["Actual_class", "P_malignant", "P_benign"]
        ].head(10)
    )

    # ---------------------------------------------------------
    # 7. Apply default threshold = 0.5
    # ---------------------------------------------------------
    results = add_threshold_predictions(results, threshold=0.5)

    print("\nPredictions at threshold = 0.5:")
    print(
        results[
            ["Actual label", "P_malignant", "Predicted_label"]
        ].head(10)
    )

    # ---------------------------------------------------------
    # 8. Compare selected thresholds
    # ---------------------------------------------------------
    print("\nMalignant prediction counts:")
    count_malignant_predictions(results)

    # ---------------------------------------------------------
    # 9. Confusion matrices
    # ---------------------------------------------------------
    print_confusion_matrices(
        probabilities,
        y_test,
        thresholds=(0.30, 0.50, 0.70)
    )

    # ---------------------------------------------------------
    # 10. Full threshold evaluation
    # ---------------------------------------------------------
    metrics_df = evaluate_thresholds(
        probabilities,
        y_test,
        thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)
    )

    print("\nThreshold evaluation:")
    print(metrics_df.round(3))


if __name__ == "__main__":
    main()
