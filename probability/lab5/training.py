import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#load the dataset
def load_prepare():
    data=load_breast_cancer()
    x=pd.DataFrame(data.data,columns=data.feature_names)
    y=pd.Series((data.target==0).astype(int),name="malignant")
    return x,y
    # print(y.value_counts())
    # print("Feature matrix shape:", x.shape)
    # print("Target shape",y.shape)
    # print("Class name:", data.target_names)
def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
