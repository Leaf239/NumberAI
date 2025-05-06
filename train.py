# train.py
import numpy as np
from model import NeuralNet
from data import load_mnist

# Load data
X, y = load_mnist()
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Hyperparameters
input_size = 784
hidden_sizes = [512,256]  # 두 개의 은닉층
output_size = 10
learning_rate = 0.001
epochs = 20
batch_size = 64

# Model (업그레이드된 버전 필요)
model = NeuralNet(input_size, hidden_sizes, output_size)

# Training loop
for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    epoch_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        preds = model.forward(X_batch)
        loss = model.compute_loss(preds, y_batch)
        grads = model.backward(X_batch, y_batch)
        model.update_params(grads, learning_rate)
        epoch_loss += loss

    avg_loss = epoch_loss / (X_train.shape[0] // batch_size)
    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

# Accuracy check
test_preds = model.forward(X_test)
pred_labels = np.argmax(test_preds, axis=1)
acc = np.mean(pred_labels == y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")