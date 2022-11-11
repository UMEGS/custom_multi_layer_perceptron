import time

import numpy as np
from IPython.core.display import clear_output


class CustomMLP():

    def __init__(self, hidden_layer_size=(1,), activations=('relu',), learning_rate=0.01, alpha=0.01, loss='binary'):
        self.n_outputs = None
        self.x = None
        self.y = None
        self.xt = None
        self.n_features = None
        self.n_samples = None
        self.batch_size = None
        self.hidden_layer_size = [0, *hidden_layer_size]
        self.activations = activations
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.loss = loss

        self.weights = []
        self.biases = []
        self.A = []

    def _init_weights_biases(self, x):

        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]


        self.hidden_layer_size[0] = self.n_features

        for i in range(1, len(self.hidden_layer_size)):
            self.weights.append(np.random.randn(self.hidden_layer_size[i], self.hidden_layer_size[i - 1]) * self.alpha)
            self.biases.append(np.random.randn(self.hidden_layer_size[i],1) * self.alpha)

    def _apply_activation(self, x, activation):
        # TODO: Implement the activation functions / derivatives / add into the class
        if activation == 'relu':
            return self._relu(x)
        elif activation == 'sigmoid':
            return self._sigmoid(x)
        elif activation == 'softmax':
            return self._softmax(x)
        elif activation == 'tanh':
            return self._tanh(x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return 1. * (x > 0)

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _softmax_derivative(self, x):
        return self._softmax(x) * (1 - self._softmax(x))

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - self._tanh(x) ** 2

    def _categorical_loss(self, y, a):
        return -np.sum(np.multiply(y, np.log(a)))

    def _categorical_loss_derivative(self, y, a):
        return -np.divide(y, a)

    def _binary_loss(self, y, a):
        return -np.sum(y * np.log() + (1 - y) * np.log(1 - a))

    def _binary_loss_derivative(self, y, a):
        return -(np.divide(y, a) - np.divide(1 - y, 1 - a))

    def _activation_derivative(self, a, activation):
        if activation == 'relu':
            return self._relu_derivative(a)
        elif activation == 'sigmoid':
            return self._sigmoid_derivative(a)
        elif activation == 'softmax':
            return self._softmax_derivative(a)
        elif activation == 'tanh':
            return self._tanh_derivative(a)

    def _find_dz(self, y, a, activation):
        if self.loss == 'binary':
            return self._binary_loss_derivative(y, a) * self._activation_derivative(a, activation)
        elif self.loss == 'categorical':
            return self._categorical_loss_derivative(y, a) * self._activation_derivative(a, activation)

    def _forward(self, x):
        a = x
        for i in range(len(self.weights)):
            Z = np.dot(self.weights[i], a) + self.biases[i]
            a = self._apply_activation(Z, self.activations[i])

            self.A.append(a)

        return a

    def _backward(self,x,y):
        dZ = self._find_dz(y, self.A[-1], self.activations[-1])
        dW = 1 / self.batch_size * np.dot(dZ, self.A[-2].T)
        db = 1 / self.batch_size * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.weights[-1].T, dZ)

        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db

        for i in reversed(range(len(self.weights) - 1)):
            dZ = dA * self._activation_derivative(self.A[i], self.activations[i])
            A = x if i == 0 else self.A[i - 1]
            dW = 1 / self.batch_size * np.dot(dZ, A.T)
            db = 1 / self.batch_size * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.weights[i].T, dZ)

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def _validate_input(self, x, y, batch_size):
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must have the same number of samples')
        if batch_size > x.shape[0]:
            raise ValueError('batch_size must be less than or equal to the number of samples')
        if batch_size < 1:
            raise ValueError('batch_size must be greater than 0')
        # if len(x.shape) != 2:
        #     raise ValueError('x must be a 2D array')
        # if len(y.shape) != 2:
        #     raise ValueError('y must be a 2D array')

    def _prepare_data(self, x, y, batch_size):
        self.x = np.array(x)
        self.y = np.array(y)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

        self.batch_size = self.n_samples if batch_size is None else batch_size
        self._validate_input(x, y, self.batch_size)  # TODO: Check if this is the right place to validate

        self.xt = self.x.T
        self.yt = self.y.T

        # self.n_outputs = self.y.shape[1]

        self._init_weights_biases(self.x)

    def fit(self, x, y, epochs=100, batch_size=32, verbose=0):

        self._prepare_data(x, y, batch_size)

        for epoch in range(epochs):
            batches = np.floor(self.n_samples / self.batch_size).astype(int)
            total_loss = 0
            b_samples = [i for i in range(0, self.n_samples, self.batch_size)]
            np.random.shuffle(b_samples)
            for i in b_samples:
                x_batch = self.x[i:i + self.batch_size]
                y_batch = self.y[i:i + self.batch_size]

                xt_batch = self.xt[:,i: i + self.batch_size]
                yt_batch = self.yt[:,i: i + self.batch_size]

                # self._forward(x_batch)
                # self._backward(x_batch,y_batch)
                batch_prediction = self._forward(xt_batch)
                self._backward(xt_batch,yt_batch)
                self.A = []

                if self.loss == 'binary':
                    total_loss += self._binary_loss(yt_batch, batch_prediction)
                elif self.loss == 'categorical':
                    total_loss += self._categorical_loss(yt_batch, batch_prediction)

                if verbose >= 1:
                    # clear_output()
                    print(f'Epoch: {epoch + 1}/{epochs}, Batch: {i}, Loss: {total_loss/(i+1)}', end="\r")

            if verbose >= 1:
                # clear_output()
                print(f'Epoch: {epoch + 1}/{epochs}, Loss: {total_loss/batches}')

    def predict(self, x):
        prediction = self._forward(x.T)
        return prediction.T

