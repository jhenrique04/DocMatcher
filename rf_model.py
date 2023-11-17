from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
import joblib
import logging

class RFModel(BaseEstimator, ClassifierMixin):
  def __init__(self, n_estimators=100, random_state=None):
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
    self.logger = logging.getLogger(__name__)

  def fit(self, X, y):
    self.logger.info("Training started")
    self.clf.fit(X, y)
    self.logger.info("Training completed")
    return self

  def predict(self, X):
    return self.clf.predict(X)

  def predict_proba(self, X):
    return self.clf.predict_proba(X)

  def get_feature_importances(self, feature_names):
    importances = self.clf.feature_importances_
    return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

  def save_model(self, path):
    joblib.dump(self.clf, path)
    self.logger.info(f"Model saved to {path}")

  def load_model(self, path):
    self.clf = joblib.load(path)
    self.logger.info(f"Model loaded from {path}")

  def tune_hyperparameters(self, X, y, param_grid):
    grid_search = GridSearchCV(self.clf, param_grid, cv=5)
    grid_search.fit(X, y)
    self.clf = grid_search.best_estimator_
    return grid_search.best_params_

  def cross_validate(self, X, y, cv=5):
    scores = cross_val_score(self.clf, X, y, cv=cv)
    self.logger.info(f"Cross-validation scores: {scores}")
    return scores

  def evaluate_model(self, X, y_true):
    y_pred = self.predict(X)
    y_proba = self.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    metrics = {
      "Confusion Matrix": confusion_matrix(y_true, y_pred),
      "ROC AUC": roc_auc_score(y_true, y_proba),
      "Precision-Recall AUC": pr_auc,
    }
    for key, value in metrics.items():
      self.logger.info(f"{key}: {value}")
    return metrics