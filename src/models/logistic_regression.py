import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X: pd.DataFrame, y: pd.Series, random_state=42):
    model = LogisticRegression(random_state=random_state)
    model.fit(X, y)
    
    return model
