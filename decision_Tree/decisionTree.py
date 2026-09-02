import numpy as np


class decisionTree:
    def __init__(
        self,
        algorithm="id3",
        max_depth=None,
        min_samples_split=2,
        min_gain=1e-12,
    ):
        """
        id3: 信息增益
        c4.5: 信息增益率
        """
        if algorithm not in ("id3", "c4.5"):
            raise ValueError("algorithm 只能是 id3 或 c4.5")

        self.tree = None
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    @staticmethod
    def entropy(label):
        _, label_counts = np.unique(label, return_counts=True)
        p = label_counts / label_counts.sum()
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _majority_label(label):
        label_classes, label_counts = np.unique(label, return_counts=True)
        return label_classes[np.argmax(label_counts)]

    def gain(self, feature, label, index):
        """计算第 index 个离散特征的信息增益。"""
        index_feature = feature[:, index]
        classes, counts = np.unique(index_feature, return_counts=True)

        H_classes = 0
        for feature_class, count in zip(classes, counts):
            choice = index_feature == feature_class
            H_classes += count / len(label) * self.entropy(label[choice])

        return self.entropy(label) - H_classes

    def gain_ratio(self, feature, label, index):
        gain = self.gain(feature, label, index)
        _, counts = np.unique(feature[:, index], return_counts=True)
        p = counts / counts.sum()
        split_info = -np.sum(p * np.log2(p))

        if split_info == 0:
            return 0
        return gain / split_info

    def get_best_feature(self, feature, label, available_features):
        best_feature = None
        max_score = -np.inf

        for index in available_features:
            if self.algorithm == "id3":
                score = self.gain(feature, label, index)
            else:
                score = self.gain_ratio(feature, label, index)

            if score > max_score:
                max_score = score
                best_feature = index

        return best_feature, max_score

    def create_tree(self, feature, label, available_features=None, depth=0):
        """ID3/C4.5 按离散特征进行多叉划分。"""
        if available_features is None:
            available_features = list(range(feature.shape[1]))

        majority = self._majority_label(label)
        if len(np.unique(label)) == 1:
            return label[0]
        if len(available_features) == 0:
            return majority
        if len(label) < self.min_samples_split:
            return majority
        if self.max_depth is not None and depth >= self.max_depth:
            return majority

        best_feature, max_score = self.get_best_feature(
            feature, label, available_features
        )
        if best_feature is None or max_score <= self.min_gain:
            return majority

        tree = {
            "feature": best_feature,
            "children": {},
            "majority": majority,
        }
        new_features = [
            index for index in available_features if index != best_feature
        ]

        for value in np.unique(feature[:, best_feature]):
            choice = feature[:, best_feature] == value
            tree["children"][value] = self.create_tree(
                feature[choice], label[choice], new_features, depth + 1
            )

        return tree

    def _predict_one(self, x, tree):
        while isinstance(tree, dict):
            best_feature = tree["feature"]
            tree = tree["children"].get(
                x[best_feature], tree["majority"]
            )
        return tree

    def _post_prune(self, tree, feature, label):
        """自底向上的 reduced-error 后剪枝。"""
        if not isinstance(tree, dict) or len(label) == 0:
            return tree

        best_feature = tree["feature"]
        for value, child in tree["children"].items():
            choice = feature[:, best_feature] == value
            tree["children"][value] = self._post_prune(
                child, feature[choice], label[choice]
            )

        tree_predict = np.array([
            self._predict_one(x, tree)
            for x in feature
        ])
        leaf_predict = np.full(len(label), tree["majority"])

        if np.mean(leaf_predict == label) >= np.mean(tree_predict == label):
            return tree["majority"]
        return tree

    def post_prune(self, feature, label):
        """使用独立验证集进行后剪枝。"""
        if self.tree is None:
            raise ValueError("请先调用 fit")

        feature, label = self._check_data(feature, label)
        self.tree = self._post_prune(self.tree, feature, label)
        return self

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


"""
"""
import numpy as np

def cal_entropy(label):
    classes, counts = np.unique(label, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def cal_gain(feature, label, index):
    feature = np.asarray(feature)
    label = np.asarray(label).reshape(-1)

    H = cal_entropy(label)
    index_feature = feature[:, index]
    values, counts = np.unique(index_feature, return_counts=True)
    p_counts = counts / counts.sum()

    for value, p in zip(values, p_counts):
        new_label = label[index_feature == value]
        H -= cal_entropy(new_label) * p

    return H

def cal_gain_ratio(feature, label, index):
    feature = np.asarray(feature)
    label = np.asarray(label).reshape(-1)

    H = cal_entropy(label)
    index_feature = feature[:, index]
    values, counts = np.unique(index_feature, return_counts=True)
    p_counts = counts / counts.sum()

    for value, p in zip(values, p_counts):
        new_label = label[index_feature == value]
        H -= cal_entropy(new_label) * p

    div = -np.sum(p_counts * np.log2(p_counts))

    if np.isclose(div, 0):
        return 0.0
    return H / div

def gini_(label):
    classes, counts = np.unique(label, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(np.square(p))

def cal_gini(feature, label, index):
    feature = np.asarray(feature)
    label = np.asarray(label).reshape(-1)

    index_feature = feature[:, index]
    values, counts = np.unique(index_feature, return_counts=True)
    p_counts = counts / counts.sum()

    gini = 0
    for value, p in zip(values, p_counts):
        new_label = label[index_feature == value]
        gini += gini_(new_label) * p
    return gini

def createTree(feature, label, available=None):
    label_classes, label_counts = np.unique(label, return_counts=True)
    max_label_index = np.argmax(label_counts)
    majority_label = label_classes[max_label_index]
    row, col = feature.shape

    if len(label_classes) == 1:
        return label_classes[0]
    if available is None:
        available = list(range(col))
    if not available or len(np.unique(feature, axis=0)) == 1:
        return majority_label

    tree = {}
    best_feature = 0
    best_gain = 0
    for index in next_available:
        gain = cal_gain(feature, label, index)
        if gain > best_gain:
            best_gain = gain
            best_feature = index
    next_available = [index for index in available if index != best_feature]

    tree[best_feature] = {}
    index_feature = feature[:, best_feature]
    values = np.unique(index_feature)
    for value in values:
        choice = index_feature == value
        sub_feature = feature[choice]
        sub_label = label[choice]
        tree[best_feature][value] = createTree(sub_feature, sub_label, next_available)
    return tree
