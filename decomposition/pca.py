import numpy as np

class pca:
    def __init__(self, d):
        self.d = d
        self.theta = None
        self.mean_feature = None

    def fit(self, feature):
        feature = np.asarray(feature, dtype=float)
        if feature.ndim != 2 or feature.shape[0] < 2:
            raise ValueError("feature 必须是至少包含两个样本的二维数组")
        if not isinstance(self.d, (int, np.integer)):
            raise ValueError("d 必须是整数")
        if not 1 <= self.d <= feature.shape[1]:
            raise ValueError("d 必须满足 1 <= d <= 特征数")

        self.mean_feature = np.mean(feature, axis=0)
        demean_feature = feature - self.mean_feature
        cov_feature = np.cov(demean_feature)
        U, S, Vh = np.linalg.svd(cov_feature)
        choice = np.argsort(S)[::-1][:self.d]
        self.theta = Vh[choice].T
        return self

    def transform(self, feature):
        if self.theta is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=float)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        if feature.ndim != 2 or feature.shape[1] != self.mean_feature.shape[0]:
            raise ValueError("feature 的特征数与训练数据不一致")

        demean_feature = feature - self.mean_feature
        return demean_feature @ self.theta
