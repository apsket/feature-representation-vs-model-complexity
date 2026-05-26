import json
import pandas as pd


class PointsDataset:
    def __init__(self, X: pd.DataFrame, y, coordinate_system=None, feature_mapping=None):
        if not isinstance(coordinate_system, (str, type(None))):
            raise ValueError("coordinate_system must be a string or None")
        if not isinstance(feature_mapping, (dict, type(None))):
            raise ValueError("feature_mapping must be a dictionary or None")
        if not isinstance(X, pd.DataFrame) or (set(X.columns) != {'x1', 'x2'}):
            raise ValueError("Input X must be a Pandas DataFrame with only 'x1' and 'x2' columns")
        
        self.X = X
        self.y = y
        self._coordinate_system = coordinate_system if coordinate_system else 'cartesian'
        self.feature_mapping = feature_mapping if feature_mapping else {'x1': 'x', 'x2': 'y'}

    @classmethod
    def from_json(cls, data_path, random_state=42, coordinate_system=None, feature_mapping=None):
        with open(data_path, 'r') as f:
            data = json.load(f)

        pos = pd.DataFrame(data['positive']).apply(pd.to_numeric)
        neg = pd.DataFrame(data['negative']).apply(pd.to_numeric)

        pos['label'] = 1
        neg['label'] = 0

        df = pd.concat([pos, neg], ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        col1, col2 = list(data['positive'][0].keys()) if 'positive' in data else list(data['negative'][0].keys())

        X = df[[col1, col2]].rename(columns={col1: 'x1', col2: 'x2'})
        y = df['label']

        return cls(X=X, y=y, coordinate_system=coordinate_system, feature_mapping=feature_mapping)

    def transform(self, transform_fn, new_coordinate_system=None, new_feature_mapping=None):
        X_transformed = transform_fn(self.X['x1'], self.X['x2'])
        
        coord_sys = new_coordinate_system or self._coordinate_system
        if (new_coordinate_system != 'cartesian') and (new_feature_mapping is None):
            raise ValueError(f"When new_coordinate_system is given, new_feature_mapping is required")
        feat_map = new_feature_mapping or self.feature_mapping
        
        return PointsDataset(X_transformed, self.y.copy(), coordinate_system=coord_sys, feature_mapping=feat_map)
    
    @property
    def semantic_X(self):
        return self.X.rename(columns=self.feature_mapping)
    
    def merge_dataset(self, other_dataset: 'PointsDataset'):
        if self.coordinate_system != other_dataset.coordinate_system:
            raise ValueError(f"PointsDatasets to merge must be of the same type. Cannot merge f{self.coordinate_system} and {other_dataset.coordinate_system}")
        self.X = pd.concat([self.X, other_dataset.X])
        self.y = pd.concat([self.y, other_dataset.y])
