from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

class SVMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='linear', probability=True):
        self.kernel = kernel
        self.probability = probability
        self.clf = SVC(kernel=self.kernel, probability=self.probability)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        if self.probability:
            return self.clf.predict_proba(X)
        else:
            raise RuntimeError("Probability estimates are not available since probability parameter is set to False")