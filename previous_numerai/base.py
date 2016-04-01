from sklearn.base import BaseEstimator

class KerasNeuralNet(BaseEstimator):
    def __init__(self, nn):
        self.nn = nn

    def fit(self, X, y):
        return self


