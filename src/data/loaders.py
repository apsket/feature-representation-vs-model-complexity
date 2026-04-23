import json
import pandas as pd

class PointsDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    @classmethod
    def from_json(cls, data_path, random_state=42):
        with open(data_path, 'r') as f:
            data = json.load(f)

        pos = pd.DataFrame(data['positive'][0]).apply(pd.to_numeric)
        neg = pd.DataFrame(data['negative'][0]).apply(pd.to_numeric)

        pos['label'] = 1
        neg['label'] = 0

        df = pd.concat([pos, neg], ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        X = df[['x', 'y']]
        y = df['label']

        return cls(X, y)

    def transform(self, transform_fn):
        X_transformed = transform_fn(self.X['x'], self.X['y'])
        return PointsDataset(X_transformed, self.y.copy())
