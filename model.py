import numpy as np

class NeuralNet:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = len(hidden_sizes) + 1
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []

        for i in range(self.layers):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.inputs = [X]
        self.zs = []

        A = X
        for i in range(self.layers - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.zs.append(Z)
            self.inputs.append(A)

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.softmax(Z)
        self.zs.append(Z)
        self.inputs.append(A)
        return A

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        log_probs = -np.log(Y_pred[range(m), Y_true] + 1e-8)
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, Y_true):
        m = X.shape[0]
        grads_W = [None] * self.layers
        grads_b = [None] * self.layers

        dZ = self.inputs[-1]
        dZ[range(m), Y_true] -= 1
        dZ /= m

        for i in reversed(range(self.layers)):
            A_prev = self.inputs[i]
            grads_W[i] = np.dot(A_prev.T, dZ)
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            if i != 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self.relu_derivative(self.zs[i-1])

        return grads_W, grads_b

    def update_params(self, grads, learning_rate):
        grads_W, grads_b = grads
        for i in range(self.layers):
            self.weights[i] -= learning_rate * grads_W[i]
            self.biases[i] -= learning_rate * grads_b[i]