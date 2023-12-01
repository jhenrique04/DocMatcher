from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
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

  def evaluate_model(self, X, y_true, cv=None):
    y_pred = self.predict(X)
    y_proba = self.clf.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    metrics = {
      "Confusion Matrix": confusion_matrix(y_true, y_pred),
      "ROC AUC": roc_auc_score(y_true, y_proba),
      "Precision-Recall AUC": pr_auc,
      "Accuracy": accuracy_score(y_true, y_pred),
    }

    if cv is not None:
      cv_scores = cross_val_score(self.clf, X, y_true, cv=cv)
      self.logger.info(f"Cross-validation scores: {cv_scores}")
      metrics["Cross-Validation Average Score"] = cv_scores.mean()

    for key, value in metrics.items():
      self.logger.info(f"{key}: {value}")
    return metrics

  def tune_hyperparameters(self, X, y, param_grid):
    grid_search = GridSearchCV(self.clf, param_grid, cv=5)
    grid_search.fit(X, y)
    self.clf = grid_search.best_estimator_
    return grid_search.best_params_
  
  def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.figure()
    plt.title(title)
    if ylim is not None:
      plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
      estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt

  def plot_validation_curve(estimator, X, y, param_name, param_range, cv=None, scoring='accuracy', n_jobs=None):
    train_scores, test_scores = validation_curve(
      estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(7, 3))
    plt.title(f'Validation Curve for {param_name}')
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label='Training score', color='darkorange', lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='darkorange', lw=lw)
    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='navy', lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='navy', lw=lw)
    plt.legend(loc='best')
    plt.show()