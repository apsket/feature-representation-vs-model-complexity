import json
import pandas as pd
import numpy as np


class PointsDataset:
    def __init__(self, X: pd.DataFrame, y: np.array):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a Pandas DataFrame")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("Input y must be a 1D NumPy array")
        
        self.X = X
        self.y = y

    @classmethod
    def from_json(cls, data_path, random_state=42):
        with open(data_path, 'r') as f:
            data = json.load(f)

        pos = pd.DataFrame(data['positive']).apply(pd.to_numeric)
        neg = pd.DataFrame(data['negative']).apply(pd.to_numeric)

        pos['label'] = 1
        neg['label'] = 0

        df = pd.concat([pos, neg], ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        X = df[[col for col in df.columns if col != 'label']]
        y = np.array(df['label'])

        return cls(X=X, y=y)

    def transform(self, transform_fn):
        X_transformed = transform_fn(self.X)
        return PointsDataset(X_transformed, self.y.copy())
    
    @classmethod
    def merge_dataset(cls, dataset1: 'PointsDataset', dataset2: 'PointsDataset'):
        return cls(
            X=pd.concat([dataset1.X, dataset2.X]),
            y=np.concat((dataset1.y, dataset2.y))
        )
