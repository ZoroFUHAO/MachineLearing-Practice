import numpy as np

class perceptron:
    def __init__(self, lr=1e-3, max_iter=100):
        self.lr = lr
        self.b = 0
        self.max_iter = max_iter

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)

        self.w = np.zeros(train_data.shape[1])

        for _ in range(self.max_iter):
            error_count = 0

            for x, y in zip(train_data, train_label):
                predict = x @ self.w + self.b

                if predict * y <= 0:
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    error_count += 1

            if not error_count:
                break

        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        predict = test_data @ self.w + self.b
        ans = np.sign(predict)
        return ans
