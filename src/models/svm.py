import pandas as pd
from sklearn.svm import SVC


def train_svm(X: pd.DataFrame, y: pd.Series):
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X, y)

    return svm_model
