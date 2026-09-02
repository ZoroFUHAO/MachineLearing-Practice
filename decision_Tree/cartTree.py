import numpy as np


class cartTree:
    """使用基尼系数的 CART 分类树，按离散特征值进行二叉划分。"""
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_gini_decrease=1e-12,
    ):
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gini_decrease = min_gini_decrease

    @staticmethod
    def gini(label):
        _, label_counts = np.unique(label, return_counts=True)
        p = label_counts / label_counts.sum()
        return 1 - np.sum(p ** 2)

    @staticmethod
    def _majority_label(label):
        label_classes, label_counts = np.unique(label, return_counts=True)
        return label_classes[np.argmax(label_counts)]

    def _split_gini(self, feature, label, index, value):
        """使用 x[index] == value 和 x[index] != value 做二叉划分。"""
        choice = feature[:, index] == value
        left_label = label[choice]
        right_label = label[~choice]

        if len(left_label) == 0 or len(right_label) == 0:
            return np.inf

        return (
            len(left_label) / len(label) * self.gini(left_label)
            + len(right_label) / len(label) * self.gini(right_label)
        )

    def get_best_split(self, feature, label):
        best_feature = None
        best_value = None
        min_gini = np.inf

        for index in range(feature.shape[1]):
            values = np.unique(feature[:, index])
            for value in values:
                score = self._split_gini(feature, label, index, value)
                if score < min_gini:
                    min_gini = score
                    best_feature = index
                    best_value = value

        return best_feature, best_value, min_gini

    def create_tree(self, feature, label, depth=0):
        majority = self._majority_label(label)
        if len(np.unique(label)) == 1:
            return label[0]
        if len(label) < self.min_samples_split:
            return majority
        if self.max_depth is not None and depth >= self.max_depth:
            return majority

        best_feature, best_value, min_gini = self.get_best_split(feature, label)
        if best_feature is None:
            return majority
        if self.gini(label) - min_gini <= self.min_gini_decrease:
            return majority

        choice = feature[:, best_feature] == best_value
        tree = {
            "feature": best_feature,
            "value": best_value,
            "majority": majority,
        }
        tree["left"] = self.create_tree(
            feature[choice], label[choice], depth + 1
        )
        tree["right"] = self.create_tree(
            feature[~choice], label[~choice], depth + 1
        )
        return tree

    def _predict_one(self, x, tree):
        while isinstance(tree, dict):
            if x[tree["feature"]] == tree["value"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return tree

    def _post_prune(self, tree, feature, label):
        """自底向上的 reduced-error 后剪枝。"""
        if not isinstance(tree, dict) or len(label) == 0:
            return tree

        best_feature = tree["feature"]
        choice = feature[:, best_feature] == tree["value"]
        tree["left"] = self._post_prune(
            tree["left"], feature[choice], label[choice]
        )
        tree["right"] = self._post_prune(
            tree["right"], feature[~choice], label[~choice]
        )

        tree_predict = np.array([
            self._predict_one(x, tree)
            for x in feature
        ])
        leaf_predict = np.full(len(label), tree["majority"])

        if np.mean(leaf_predict == label) >= np.mean(tree_predict == label):
            return tree["majority"]
        return tree

    @staticmethod
    def _check_data(feature, label=None):
        feature = np.asarray(feature)
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

    def post_prune(self, feature, label):
        """使用独立验证集进行后剪枝。"""
        if self.tree is None:
            raise ValueError("请先调用 fit")

        feature, label = self._check_data(feature, label)
        self.tree = self._post_prune(self.tree, feature, label)
        return self

    def fit(
        self,
        feature,
        label,
        validation_feature=None,
        validation_label=None,
    ):
        feature, label = self._check_data(feature, label)
        self.tree = self.create_tree(feature, label)

        if validation_feature is not None or validation_label is not None:
            if validation_feature is None or validation_label is None:
                raise ValueError("后剪枝需要同时提供验证集特征和标签")
            self.post_prune(validation_feature, validation_label)

        return self

    def predict(self, feature):
        if self.tree is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        if feature.ndim != 2:
            raise ValueError("feature 必须是二维数组")

        return np.array([self._predict_one(x, self.tree) for x in feature])
