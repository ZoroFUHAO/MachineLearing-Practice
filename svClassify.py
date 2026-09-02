import numpy as np

class svc:
    def __init__(self, C, max_iter=500, learning_rate=1e-3):
        self.C = C
        self.max_iter = max_iter
        self.lr = learning_rate
        self.w = None
        self.b = 0.0

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)

        self.w = np.zeros(train_data.shape[1])
        self.b = 0.0
        self.classes = np.unique(train_label)
        label = np.where(train_label == self.classes[0], 1.0, -1.0)

        for _ in range(self.max_iter):
            scores = train_data @ self.w + self.b
            margins = label * scores
            violating = margins < 1

            gradient_w = self.w.copy()
            gradient_b = 0.0
            # if np.any(violating):
            #     gradient_w -= (self.C / train_data.shape[0]) * (train_data[violating].T @ label[violating])
            #     gradient_b = -self.C / train_data.shape[0] * np.sum(label[violating])
            for i in range(train_data.shape[0]):
                if violating[i]:
                    gradient_w -= self.C / train_data.shape[0] * label[i] * train_data[i]
                    gradient_b -= self.C / train_data.shape[0] * label[i]

            self.w -= self.lr * gradient_w
            self.b -= self.lr * gradient_b

        final_scores = label * (train_data @ self.w + self.b)
        self.support_vector = train_data[final_scores <= 1 + 1e-6]
        return self

    def calculate_scores(self, X):
        return X @ self.w + self.b

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        scores = self.calculate_scores(test_data)
        return np.where(scores >= 0, self.classes[0], self.classes[1])
    
            