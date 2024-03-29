import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        _y = np.where(y <=0 , -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = _y[idx] * (np.dot(x_i, self.weights) - self.bias) >=1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param *self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param *self.weights - np.dot(x_i, _y[idx]))
                    self.bias -= self.learning_rate * _y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)