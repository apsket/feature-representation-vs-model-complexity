from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd


class ModelMetrics():
    def __init__(self, model_name, dataset_name, response, estimated_response):
        self.response = response
        self.estimated_response = estimated_response
        self.metrics = {'model': model_name, 'dataset': dataset_name}
    
    def compute_metrics(self):
        metrics = {
            'accuracy': accuracy_score(self.response, self.estimated_response),
            'precision': precision_score(self.response, self.estimated_response),
            'recall': recall_score(self.response, self.estimated_response),
            'f1': f1_score(self.response, self.estimated_response)
        }

        self.metrics.update(metrics)

        self.confusion_matrix = confusion_matrix(self.response, self.estimated_response)
    
    def to_dict(self):
        return {key: value for key, value in self.metrics.items()}

    def get_confusion_matrix(self):
        return self.confusion_matrix
    
    def to_pandas(self):
        return pd.DataFrame([self.metrics])
