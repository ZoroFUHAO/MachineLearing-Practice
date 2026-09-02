import numpy as np


class mlp:
    """单隐藏层 MLP：ReLU + Softmax，用于二分类和多分类。"""
    def __init__(
        self,
        hidden_size=16,
        learning_rate=1e-2,
        max_iter=1000,
        seed=42,
    ):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.classes_ = None
        self.loss = []

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def _check_data(feature, label=None):
        feature = np.asarray(feature, dtype=float)
        if feature.ndim != 2:
            raise ValueError("feature 必须是二维数组")

        if label is None:
            return feature

        label = np.asarray(label).reshape(-1)
        if len(feature) != len(label):
            raise ValueError("feature 和 label 的样本数必须一致")
        if len(label) == 0:
            raise ValueError("训练数据不能为空")
        return feature, label

    def _init_params(self, feature_count, class_count):
        # ReLU 使用 He 初始化，避免隐藏层输出过小。
        self.W1 = self.rng.normal(
            0, np.sqrt(2 / feature_count), size=(feature_count, self.hidden_size)
        )
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = self.rng.normal(
            0, np.sqrt(2 / self.hidden_size), size=(self.hidden_size, class_count)
        )
        self.b2 = np.zeros(class_count)

    def fit(self, feature, label):
        feature, label = self._check_data(feature, label)
        if not isinstance(self.hidden_size, (int, np.integer)) or self.hidden_size < 1:
            raise ValueError("hidden_size 必须是正整数")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter 必须是正整数")
        if self.lr <= 0:
            raise ValueError("learning_rate 必须大于 0")

        self.classes_ = np.unique(label)
        if len(self.classes_) < 2:
            raise ValueError("至少需要两个类别")

        # 将任意类别标签转成 0 到 C - 1 的下标。
        label_index = np.searchsorted(self.classes_, label)
        one_hot = np.eye(len(self.classes_))[label_index]
        self._init_params(feature.shape[1], len(self.classes_))
        self.loss = []

        for _ in range(self.max_iter):
            # 前向传播
            hidden_linear = feature @ self.W1 + self.b1
            hidden = self.relu(hidden_linear)
            score = hidden @ self.W2 + self.b2
            prob = self.softmax(score)

            loss = -np.mean(np.sum(one_hot * np.log(prob + 1e-12), axis=1))
            self.loss.append(float(loss))

            # 反向传播
            d_score = (prob - one_hot) / feature.shape[0]
            d_W2 = hidden.T @ d_score
            d_b2 = np.sum(d_score, axis=0)

            d_hidden = d_score @ self.W2.T
            d_hidden[hidden_linear <= 0] = 0
            d_W1 = feature.T @ d_hidden
            d_b1 = np.sum(d_hidden, axis=0)

            # 梯度下降
            self.W1 -= self.lr * d_W1
            self.b1 -= self.lr * d_b1
            self.W2 -= self.lr * d_W2
            self.b2 -= self.lr * d_b2

        return self

    def predict_proba(self, feature):
        if self.W1 is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=float)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        feature = self._check_data(feature)
        if feature.shape[1] != self.W1.shape[0]:
            raise ValueError("feature 的特征数与训练数据不一致")

        hidden = self.relu(feature @ self.W1 + self.b1)
        return self.softmax(hidden @ self.W2 + self.b2)

    def predict(self, feature):
        prob = self.predict_proba(feature)
        return self.classes_[np.argmax(prob, axis=1)]
