from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

class RFModel(BaseEstimator, ClassifierMixin):
  def __init__(self, n_estimators=100, random_state=None):
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
    
  def fit(self, X, y):
    self.clf.fit(X, y)
    return self

  def predict(self, X):
    return self.clf.predict(X)

  def predict_proba(self, X):
    return self.clf.predict_proba(X)