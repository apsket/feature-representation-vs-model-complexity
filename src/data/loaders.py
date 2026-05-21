import json
import pandas as pd


class PointsDataset:
    def __init__(self, X, y, coordinate_system=None, feature_mapping=None):
        if not isinstance(coordinate_system, (str, type(None))):
            raise ValueError("coordinate_system must be a string or None")
        if not isinstance(feature_mapping, (dict, type(None))):
            raise ValueError("feature_mapping must be a dictionary or None")
        
        self.X = X
        self.y = y
        self._coordinate_system = coordinate_system if coordinate_system else 'cartesian'
        self.feature_mapping = feature_mapping if feature_mapping else {'x': 'x', 'y': 'y'}

    @classmethod
    def from_json(cls, data_path, random_state=42, coordinate_system=None, feature_mapping=None):
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

        return cls(X, y, coordinate_system=coordinate_system, feature_mapping=feature_mapping)

    def transform(self, transform_fn, new_coordinate_system=None, new_feature_mapping=None):
        X_transformed = transform_fn(self.X['x'], self.X['y'])
        
        coord_sys = new_coordinate_system or self._coordinate_system
        feat_map = new_feature_mapping or self.feature_mapping
        
        return PointsDataset(X_transformed, self.y.copy(), coordinate_system=coord_sys, feature_mapping=feat_map)
    
    @property
    def semantic_X(self):
        return self.X.rename(columns=self.feature_mapping)
