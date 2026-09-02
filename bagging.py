import numpy as np
from collections import Counter
from practice.classify.decisionTree import decisionTree

class BaggingClassifier:
    def __init__(self, n_model=10):
        self.n_model = n_model
        self.models = []

    def fit(self, feature, label):
        feature = np.asarray(feature)
        label = np.asarray(label)

        if len(feature) != len(label):
            raise ValueError("feature 和 label 的样本数必须一致")

        self.models = []  # 防止重复调用 fit 时保留旧模型
        n_samples = len(feature)

        for _ in range(self.n_model):
            # 有放回抽取 n_samples 个样本下标
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True
            )

            model = decisionTree()
            model.fit(feature[indices], label[indices])

            self.models.append(model)
        return self

    def predict(self, feature):
        if not self.models:
            raise ValueError("请先调用 fit() 训练模型")

        feature = np.asarray(feature)

        # shape: (模型数, 测试样本数)
        all_predictions = np.array([
            model.predict(feature)
            for model in self.models
        ])

        result = []

        # 每个样本由所有模型进行多数投票
        for predictions in all_predictions.T:
            label = Counter(predictions).most_common(1)[0][0]
            result.append(label)

        return np.array(result)