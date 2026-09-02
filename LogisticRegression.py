import numpy as np

class logisticRegression:
    def __init__(self, lr=1e-3, max_iter=100):
        self.lr = lr
        self.b = 0
        self.max_iter = max_iter

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)

        self.w = np.zeros(train_data.shape[1])
        self.loss = []

        for _ in range(self.max_iter):
            predict = self.sigmoid(train_data @ self.w + self.b)
            error = predict - train_label

            gradient_w = train_data.T @ error / train_data.shape[0]
            gradient_b = error.mean()

            self.w -= self.lr * gradient_w
            self.b -= self.lr * gradient_b
            self.loss.append(-np.mean(train_label * np.log(predict) + (1-train_label) * np.log(1-predict)))

        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        predict = self.sigmoid(test_data @ self.w + self.b)
        ans = np.where(predict >= 0.5, 1, 0)
        return ans

    def predict_proba(self, test_data):
        test_data = np.asarray(test_data)
        predict = self.sigmoid(test_data @ self.w + self.b)
        return predict