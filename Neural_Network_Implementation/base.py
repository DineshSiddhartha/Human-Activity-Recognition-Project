import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01,
                 activation='relu', reg_lambda=0.0, use_momentum=False, momentum_beta=0.9):
        self.lr = learning_rate
        self.activation_name = activation
        self.reg_lambda = reg_lambda
        self.use_momentum = use_momentum
        self.momentum_beta = momentum_beta

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        if use_momentum:
            self.v_W1 = np.zeros_like(self.W1)
            self.v_b1 = np.zeros_like(self.b1)
            self.v_W2 = np.zeros_like(self.W2)
            self.v_b2 = np.zeros_like(self.b2)

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)

    def tanh(self, x): return np.tanh(x)
    def tanh_derivative(self, x): return 1 - np.tanh(x) ** 2

    def leaky_relu(self, x): return np.where(x > 0, x, 0.01 * x)
    def leaky_relu_derivative(self, x): return np.where(x > 0, 1, 0.01)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def activate(self, x):
        if self.activation_name == 'relu': return self.relu(x)
        if self.activation_name == 'tanh': return self.tanh(x)
        if self.activation_name == 'leaky_relu': return self.leaky_relu(x)
        raise ValueError("Unsupported activation")

    def activate_derivative(self, x):
        if self.activation_name == 'relu': return self.relu_derivative(x)
        if self.activation_name == 'tanh': return self.tanh_derivative(x)
        if self.activation_name == 'leaky_relu': return self.leaky_relu_derivative(x)
        raise ValueError("Unsupported activation")

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activate(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = X.shape[0]
        y_pred = self.a2.copy()

        dz2 = y_pred
        dz2[range(m), y_true] -= 1
        dz2 /= m

        dW2 = self.a1.T @ dz2 + self.reg_lambda * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activate_derivative(self.z1)
        dW1 = X.T @ dz1 + self.reg_lambda * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        if self.use_momentum:
            self.v_W1 = self.momentum_beta * getattr(self, "v_W1", 0) + (1 - self.momentum_beta) * dW1
            self.v_b1 = self.momentum_beta * getattr(self, "v_b1", 0) + (1 - self.momentum_beta) * db1
            self.v_W2 = self.momentum_beta * getattr(self, "v_W2", 0) + (1 - self.momentum_beta) * dW2
            self.v_b2 = self.momentum_beta * getattr(self, "v_b2", 0) + (1 - self.momentum_beta) * db2

            self.W1 -= self.lr * self.v_W1
            self.b1 -= self.lr * self.v_b1
            self.W2 -= self.lr * self.v_W2
            self.b2 -= self.lr * self.v_b2
        else:
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=20, batch_size=64):
        for epoch in range(epochs):
            perm = np.random.permutation(X.shape[0])
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
