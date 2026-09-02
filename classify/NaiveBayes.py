import numpy as np


class naiveBayes(object):
    """离散特征朴素贝叶斯，使用拉普拉斯平滑。"""
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")

        self.alpha = alpha
        self.label_prob = {}
        self.condition_prob = {}
        self.feature_values = {}
        self.classes_ = None
        self.class_count = {}

    @staticmethod
    def _check_data(feature, label=None):
        feature = np.asarray(feature, dtype=object)
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

    def fit(self, feature, label):
        feature, label = self._check_data(feature, label)
        self.label_prob = {}
        self.condition_prob = {}
        self.feature_values = {} # 第j个特征的取值
        self.class_count = {}

        self.classes_, label_counts = np.unique(label, return_counts=True)
        for lc, count in zip(self.classes_, label_counts):
            self.label_prob[lc] = count / len(label)
            self.class_count[lc] = count

        for i in range(feature.shape[1]):
            self.feature_values[i] = np.unique(feature[:, i])

        for lc in self.classes_:
            self.condition_prob[lc] = {}
            class_feature = feature[label == lc]

            for i, values in self.feature_values.items():
                self.condition_prob[lc][i] = {}
                bottom = len(class_feature) + self.alpha * len(values)

                for value in values:
                    count = np.sum(class_feature[:, i] == value)
                    self.condition_prob[lc][i][value] = (
                        count + self.alpha
                    ) / bottom

        return self

    def _log_prob(self, x, lc):
        prob = np.log(self.label_prob[lc])
        for i, value in enumerate(x):
            condition = self.condition_prob[lc][i]
            if value in condition:
                prob += np.log(condition[value])
            else:
                # 未知取值单独占一个平滑概率。
                bottom = self.class_count[lc] + self.alpha * (
                    len(self.feature_values[i]) + 1
                )
                prob += np.log(self.alpha / bottom)
        return prob

    def predict(self, feature):
        if self.classes_ is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=object)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        feature = self._check_data(feature)

        result = []
        for x in feature:
            probs = [self._log_prob(x, lc) for lc in self.classes_]
            result.append(self.classes_[np.argmax(probs)])
        return np.array(result)


class GaussNaiveBayes:
    """
    混合特征朴素贝叶斯。
    continuous_features 中的列使用高斯分布，其余列按离散特征处理。
    """
    def __init__(self, continuous_features=None, alpha=1.0, var_smoothing=1e-9):
        if alpha <= 0:
            raise ValueError("alpha 必须大于 0")
        if var_smoothing <= 0:
            raise ValueError("var_smoothing 必须大于 0")

        self.continuous_features = continuous_features
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.label_prob = {}
        self.class_count = {}
        self.condition_prob = {}
        self.gauss_prob = {}
        self.feature_values = {}
        self.continuous_index = set()

    @staticmethod
    def _check_data(feature, label=None):
        feature = np.asarray(feature, dtype=object)
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

    def fit(self, feature, label):
        feature, label = self._check_data(feature, label)
        feature_count = feature.shape[1]

        if self.continuous_features is None:
            self.continuous_index = set(range(feature_count))
        else:
            self.continuous_index = set(self.continuous_features)
            if any(i < 0 or i >= feature_count for i in self.continuous_index):
                raise ValueError("continuous_features 中存在无效列下标")

        self.label_prob = {}
        self.class_count = {}
        self.condition_prob = {}
        self.gauss_prob = {}
        self.feature_values = {}
        self.classes_, label_counts = np.unique(label, return_counts=True)

        for lc, count in zip(self.classes_, label_counts):
            self.label_prob[lc] = count / len(label)
            self.class_count[lc] = count
            self.condition_prob[lc] = {}
            self.gauss_prob[lc] = {}

        for i in range(feature_count):
            if i not in self.continuous_index:
                self.feature_values[i] = np.unique(feature[:, i])

        for lc in self.classes_:
            class_feature = feature[label == lc]

            for i in range(feature_count):
                if i in self.continuous_index:
                    values = np.asarray(class_feature[:, i], dtype=float)
                    mean = values.mean()
                    var = max(values.var(), self.var_smoothing)
                    self.gauss_prob[lc][i] = (mean, var)
                else:
                    values = self.feature_values[i]
                    bottom = len(class_feature) + self.alpha * len(values)
                    self.condition_prob[lc][i] = {}

                    for value in values:
                        count = np.sum(class_feature[:, i] == value)
                        self.condition_prob[lc][i][value] = (
                            count + self.alpha
                        ) / bottom

        return self

    def _log_gauss_prob(self, value, mean, var):
        value = float(value)
        return -0.5 * (
            np.log(2 * np.pi * var) + (value - mean) ** 2 / var
        )

    def _log_prob(self, x, lc):
        prob = np.log(self.label_prob[lc])

        for i, value in enumerate(x):
            if i in self.continuous_index:
                mean, var = self.gauss_prob[lc][i]
                prob += self._log_gauss_prob(value, mean, var)
            else:
                condition = self.condition_prob[lc][i]
                if value in condition:
                    prob += np.log(condition[value])
                else:
                    bottom = self.class_count[lc] + self.alpha * (
                        len(self.feature_values[i]) + 1
                    )
                    prob += np.log(self.alpha / bottom)

        return prob

    def predict(self, feature):
        if self.classes_ is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=object)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        feature = self._check_data(feature)

        result = []
        for x in feature:
            probs = [self._log_prob(x, lc) for lc in self.classes_]
            result.append(self.classes_[np.argmax(probs)])
        return np.array(result)

    def predict_proba(self, feature):
        if self.classes_ is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=object)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        feature = self._check_data(feature)

        result = []
        for x in feature:
            log_probs = np.array([
                self._log_prob(x, lc)
                for lc in self.classes_
            ])
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            result.append(probs / probs.sum())
        return np.array(result)
