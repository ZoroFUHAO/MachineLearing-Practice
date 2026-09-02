import numpy as np

class LDA:
    """多分类 Fisher LDA：训练后将数据投影到至多 C - 1 维。"""

    def __init__(self, n_components=None, reg=1e-6):
        self.n_components = n_components
        self.reg = reg
        self.classes_ = None
        self.mean_ = None
        self.class_means_ = None
        self.components_ = None  # 形状：(特征数, 降维后的维数)

    def fit(self, train_data, train_label):
        X = np.asarray(train_data, dtype=float)
        y = np.asarray(train_label).reshape(-1)

        sample_count, feature_count = X.shape
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_count = len(self.classes_)


        self.mean_ = X.mean(axis=0)
        # 2. 每个类别的均值 μ_c，形状为 (C, d)
        self.class_means_ = np.array([
            X[y == label].mean(axis=0)
            for label in self.classes_
        ])

        # 3. 类内散度矩阵 Sw
        S_w = np.zeros((feature_count, feature_count))
        # 4. 类间散度矩阵 Sb
        S_b = np.zeros((feature_count, feature_count))

        for label, count, class_mean in zip(
            self.classes_, class_counts, self.class_means_
        ):
            class_samples = X[y == label]

            # 当前类别内部，每个样本相对于本类均值的偏移
            centered = class_samples - class_mean
            S_w += centered.T @ centered

            # 当前类别均值相对于全局均值的偏移
            mean_difference = class_mean - self.mean_
            S_b += count * np.outer(mean_difference, mean_difference)

        # LDA 最多只能得到 C - 1 个有效投影方向
        max_components = min(feature_count, class_count - 1)
        n_components = (
            max_components
            if self.n_components is None
            else min(self.n_components, max_components)
        )

        # 防止 Sw 奇异；尤其常见于“特征数 > 样本数”。
        S_w += self.reg * np.eye(feature_count)

        # 解 Sw^(-1) Sb 的特征值问题；solve 比 inv 更稳定。
        eigen_values, eigen_vectors = np.linalg.eig(
            np.linalg.solve(S_w, S_b)
        )

        # 数值误差可能产生极小虚部，这里取实部。
        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real

        # 特征向量按对应特征值从大到小排列。
        order = np.argsort(eigen_values)[::-1]

        # 注意：特征向量在“列”中，不是行中。
        self.components_ = eigen_vectors[:, order[:n_components]]

        return self

    def transform(self, test_data):
        X = np.asarray(test_data, dtype=float)

        # (样本数, 特征数) @ (特征数, 降维维数)
        return X @ self.components_

    def fit_transform(self, train_data, train_label):
        return self.fit(train_data, train_label).transform(train_data)

def lda(X, y):
    '''
    input:X(ndarray):待处理数据
        y(ndarray):待处理数据标签，标签分别为0和1
    output:X_new(ndarray):处理后的数据
    '''
    #********* Begin *********#
    #划分出第一类样本与第二类样本
    y = y.reshape(-1)
    X_1 = X[y==0]
    X_2 = X[y==1]
    #获取第一类样本与第二类样本中心点
    mean_1 = np.mean(X_1,axis=0)
    mean_2 = np.mean(X_2,axis=0)
    #计算第一类样本与第二类样本协方差矩阵
    cov_1 = (X_1-mean_1).T @ (X_1-mean_1)
    cov_2 = (X_2-mean_2).T @ (X_2-mean_2)
    #计算类内散度矩阵
    S_w = cov_1+cov_2
    #计算w
    w = np.linalg.solve(S_w, mean_1-mean_2)
    #计算新样本集
    X_new = X @ w
    #********* End *********#
    return X_new

"""
多分类
"""
def fit(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    row, col = X.shape

    y_classes, y_count = np.unique(y, return_counts=True)
    C = len(y_classes)

    mu = np.array([np.mean(X[y == c], axis=0) for c in y_classes])
    mu_all = np.mean(X, axis=0)

    S_b = np.zeros((col, col))
    S_w = np.zeros((col, col))
    for i, c in enumerate(y_classes):
        diff = mu[i]-mu_all
        S_b += y_count[i] * np.outer(diff, diff)
    
        feature = X[y == c]
        diff = feature - mu[i]
        S_w += diff.T @ diff

    S_w += 1e-6 * np.eye(col)

    W = np.linalg.solve(S_w, S_b)
    eigen_vals, eigen_vecs = np.linalg.eig(W)
    order = np.argsort(eigen_vals.real)[::-1]
    n = min(C-1, col)
    return eigen_vecs[:, order[:n]].real


