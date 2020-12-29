# -*- coding: utf-8 -*-
"""KNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dgEZhXGUWS2ZlP5zAx7iNf7WSN_XOAQH
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
  return np.sqrt(np.sum(x1-x2)**2)

class KNN:
  def __init__(self, k=3):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y  

  def predict(self, X):
    predicted_labels = [self._predict(x) for x in X]
    return np.array(predicted_labels)

  def _predict(self, x):
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common

knn = KNN()
knn.fit(X_train, y_train)
knn.predict(X_test)