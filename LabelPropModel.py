from sklearn.semi_supervised import LabelPropagation
import numpy as np

class LabelPropModel:
  def __init__(self):
    self.clf = LabelPropagation()
    
  def train(self, X, y):
    self.clf.fit(X, y)
    
  def predict(self, X):
    return self.clf.predict(X)