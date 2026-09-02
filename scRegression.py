import numpy as np

class svr:
    def __init__(self, C, max_iter=500, learning_rate=1e-3, epsilon=0.1):
        self.C = C
        self.max_iter = max_iter
        self.lr = learning_rate
        self.epsilon = epsilon
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        
        for _ in range(self.max_iter):
            predict = X @ self.w + self.b
            residual = predict - y
            outside = np.abs(residual) > self.epsilon
            sign = np.sign(residual[outside])

            gradient_w = self.w.copy()
            gradient_b = 0.0
            if np.any(outside):
                gradient_w += (self.C / X.shape[0]) * (X[outside].T @ sign)
                gradient_b = self.C / X.shape[0] * np.sum(sign)

            self.w -= gradient_w * self.lr
            self.b -= gradient_b * self.lr

        final_residual = np.abs(X @ self.w + self.b - y)
        self.support_vector = X[final_residual >= self.epsilon - 1e-6]
        return self
    
    def predict(self, X):
        return X @ self.w + self.b