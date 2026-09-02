import numpy as np

class linearRegression:
    """
    梯度下降
    """
    def __init__(self, max_iter=100, learning_rate=1e-3):
        self.w = None
        self.b = 0
        self.max_iter = max_iter
        self.lr = learning_rate

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)
        self.w = np.zeros(train_data.shape[1])
        self.loss = []

        for _ in range(self.max_iter):
            update_label = train_data @ self.w + self.b
            error = update_label - train_label

            gradient_w = 2 / train_data.shape[0] * (train_data.T @ error)
            gradient_b = 2 / train_data.shape[0] * error.mean()

            self.w -= self.lr * gradient_w
            self.b -= self.lr * gradient_b

            update_label = train_data @ self.w + self.b

            self.loss.append(float(np.mean(error ** 2)))

        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        return test_data @ self.w + self.b

class LinearRegression:
    """
    闭式解(X^\top X)^{-1} X^\top y
    """
    def __init__(self):
        self.theta = None

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)

        b_col = np.ones((train_data.shape[0],1))
        X = np.concatenate([b_col, train_data], axis=1)
        left = np.linalg.inv(X.T @ X)
        self.theta = left @ X.T @ train_label
        # self.theta = np.linalg.pinv(X) @ train_label

        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        b_col = np.ones((test_data.shape[0],1))

        X = np.concatenate([b_col, test_data], axis=1)
        return X @ self.theta
