import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, recall_score,precision_score
import matplotlib.pyplot as plt
def distribution(y):
    class_counts=y.value_counts().sort_index()
    class_distribution=pd.DataFrame({
        "class":["Benign","Malignant"],
        "Count": class_counts.values,
        "probability": class_counts.values/len(y)
        })
    return class_distribution

def plot(y):
    class_counts=y.value_counts().sort_index()
    class_distribution=pd.DataFrame({
        "class":["Benign","Malignant"],
         "Count": class_counts.values})
    class_distribution.plot(
    x="class",
    y="Count",kind="bar",
    legend=False,
    color=["orange","blue"]
)
    plt.ylabel("Number of observation")
    plt.xlabel("Class Distribution")
    plt.xticks(rotation=0)
    plt.show()
def evaluate(y_test,probabalities): 
    actual_malignant=(y_test.values==0).astype(int)

    for threshold in [0.30,0.50,0.70]:
        predicted_malignant=(probabalities[:,1]>=threshold).astype(int)
        cm=confusion_matrix(actual_malignant,predicted_malignant)
        print(f"\n Threshold={threshold} | Predicted Malignant: {predicted_malignant.sum()}")
        print("Confusion Matrix")
        print(cm)

def standard_metrics(y_test,y_pred):
    print("Sklearn maccuracy:", accuracy_score(y_test,y_pred))
    print("Sklearn f1 score:", f1_score(y_test,y_pred))
    print("Sklearn recall:", recall_score(y_test,y_pred))
    print("Sklearn precision:", precision_score(y_test,y_pred))
