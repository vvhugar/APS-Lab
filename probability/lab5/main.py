# Import dedicated functions from your structural module layers
from training import load_prepare, split_data
from cancer_model import train_pipeline
from metrics import distribution, evaluate, standard_metrics, plot
def main():
    x, y = load_prepare()
    print(f"Feature matrix shape: {x.shape}")
    print(f"Target shape:         {y.shape}")
    distribution(y)
    plot(y)
    x_train, x_test, y_train, y_test = split_data(x, y)

    pipeline_model = train_pipeline(x_train, y_train)
    
    y_pred = pipeline_model.predict(x_test)
    probabilities = pipeline_model.predict_proba(x_test)
    
    evaluate(y_test, probabilities)
    standard_metrics(y_test, y_pred)

if __name__ == "__main__":
    main()
