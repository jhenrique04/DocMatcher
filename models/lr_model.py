from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class LRModel(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, solver='lbfgs', max_iter=100):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.clf = LogisticRegression(C=self.C, solver=self.solver, max_iter=self.max_iter)
    
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self  # Fit should return "self" to allow method chaining
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)