import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

X,y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

import numpy as np

class NaiveBayes:
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self._classes = np.unique(y)
    n_classes = len(self._classes)
    self._means = np.zeros((n_classes, n_features), dtype=np.float64)
    self._vars = np.zeros((n_classes, n_features), dtype=np.float64)
    self._priors = np.zeros(n_classes, dtype=np.float64)

    for c in self._classes:
      X_c = X[y==c]
      self._means[c,:] = X_c.mean(axis = 0)
      self._vars[c,:] = X_c.var(axis = 0)
      self._priors[c] = X_c.shape[0] / float(n_samples)

  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return y_pred

  def _predict(self, x):
    posteriors = []

    for idx, c in enumerate(self._classes):
      prior = np.log(self._priors[idx])
      class_conditional = np.sum(np.log(self._pdf(idx, x)))
      posterior = prior + class_conditional
      posteriors.append(posterior)

    return self._classes[np.argmax(posteriors)]

  def _pdf(self, class_idx, x):
      mean = self._means[class_idx]
      var = self._vars[class_idx]
      numerator = np.exp(-(x-mean)**2/ (2*var))
      denominator = np.sqrt(2 * np.pi * var)
      return numerator / denominator

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print(accuracy(y_test, predictions))
