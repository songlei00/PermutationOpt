import numpy as np
from scipy.stats import norm

class EI:
    def __init__(self, model, train_X, train_y):
        self.model = model
        self.train_X_np = np.vstack(train_X)
        self.train_y_np = np.vstack(train_y)
    
    def __call__(self, x, xi=0.0001):
        assert x.ndim == 2
        mu, var = self.model.predict(x)
        sigma = np.sqrt(var)
        
        train_pred, _ = self.model.predict(self.train_X_np)
        best_f = np.max(train_pred)
        imp = mu - best_f - xi
        assert imp.shape[1] == 1
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
        return ei
        