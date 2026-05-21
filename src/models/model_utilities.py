# Wrapper so the model only looks at a subset of variables during prediction
class Model1DWrapper:
    def __init__(self, trained_model, vars_list):
        self.model = trained_model
        self.vars_list = vars_list
        
    def predict(self, X):
        # Even if passed a 2D domain dataframe, only feed the subset variables to the model
        return self.model.predict(X[self.vars_list])