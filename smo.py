import numpy as np


class smo:
    """
    线性软间隔 SVM 的对偶问题，使用 SMO 求解。

    max  sum(alpha_i) - 1/2 * sum(alpha_i * alpha_j * y_i * y_j * K_ij)
    s.t. sum(alpha_i * y_i) = 0, 0 <= alpha_i <= C
    """
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3, eps=1e-5, seed=42):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.rng = np.random.default_rng(seed)

        self.alpha = None
        self.b = 0.0
        self.w = None
        self.classes_ = None
        self.support_vector = None
        self.support_vector_index = None

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
        if len(label) < 2:
            raise ValueError("至少需要两个样本")
        return feature, label

    def _select_j(self, i, E_i, errors):
        """按 |E_i - E_j| 从大到小返回候选 j。"""
        diff = np.abs(E_i - errors)
        diff[i] = -1
        return [j for j in np.argsort(diff)[::-1] if j != i]

    def fit(self, train_data, train_label):
        train_data, train_label = self._check_data(train_data, train_label)
        if self.C <= 0:
            raise ValueError("C 必须大于 0")
        if self.max_iter < 1:
            raise ValueError("max_iter 必须大于 0")

        self.classes_ = np.unique(train_label)
        if len(self.classes_) != 2:
            raise ValueError("SMO 只支持二分类")

        # 与 svClassify.py 保持一致：第一个类别映射为 +1。
        label = np.where(train_label == self.classes_[0], 1.0, -1.0)
        sample_count = train_data.shape[0]

        # 线性核 K_ij = x_i^T x_j，先计算好避免重复计算。
        kernel = train_data @ train_data.T
        self.alpha = np.zeros(sample_count)
        self.b = 0.0

        for iteration in range(self.max_iter):
            changed_count = 0

            for i in range(sample_count):
                # 第 1 步：计算 E_i = f(x_i) - y_i。
                scores = kernel @ (self.alpha * label) + self.b
                errors = scores - label
                E_i = errors[i]

                # 第 2 步：检查 i 是否违反 KKT 条件。
                # alpha_i = 0: y_i*f_i >= 1
                # 0 < alpha_i < C: y_i*f_i = 1
                # alpha_i = C: y_i*f_i <= 1
                violate = (
                    (label[i] * E_i < -self.tol and self.alpha[i] < self.C)
                    or (label[i] * E_i > self.tol and self.alpha[i] > 0)
                )
                if not violate:
                    continue

                # 第 3 步：优先选择 |E_i-E_j| 大的 j；若不可更新，再尝试其他 j。
                for j in self._select_j(i, E_i, errors):
                    E_j = errors[j]
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # 第 4 步：由 sum(alpha_i*y_i)=0 得到 alpha_j 的上下界 L、H。
                    if label[i] != label[j]: # alpha_1 - alpha_2 = y1 * \epsilon
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else: # alpha_1 + alpha_2 = y1 * \epsilon
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    # 第 5 步：计算 eta 并更新 alpha_j。
                    eta = (
                        2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                    )
                    if eta >= 0:
                        continue

                    alpha_j_new = alpha_j_old - label[j] * (E_i - E_j) / eta
                    alpha_j_new = np.clip(alpha_j_new, L, H)

                    if abs(alpha_j_new - alpha_j_old) < self.eps:
                        continue

                    # 第 6 步：由等式约束更新 alpha_i。
                    alpha_i_new = alpha_i_old + label[i] * label[j] * (
                        alpha_j_old - alpha_j_new
                    )

                    # 第 7 步：利用 b_1、b_2 更新偏置 b。
                    b1 = (
                        self.b - E_i
                        - label[i] * (alpha_i_new - alpha_i_old) * kernel[i, i]
                        - label[j] * (alpha_j_new - alpha_j_old) * kernel[i, j]
                    )
                    b2 = (
                        self.b - E_j
                        - label[i] * (alpha_i_new - alpha_i_old) * kernel[i, j]
                        - label[j] * (alpha_j_new - alpha_j_old) * kernel[j, j]
                    )

                    if self.eps < alpha_i_new < self.C - self.eps:
                        self.b = b1
                    elif self.eps < alpha_j_new < self.C - self.eps:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new
                    changed_count += 1
                    break

            self.n_iter_ = iteration + 1
            # 一整轮都没有变量更新，说明当前解已满足 tol 范围内的 KKT。
            if changed_count == 0:
                break

        # 线性核下可以从对偶变量还原原始问题的 w。
        self.w = (self.alpha * label) @ train_data
        self.support_vector_index = np.where(self.alpha > self.eps)[0]
        self.support_vector = train_data[self.support_vector_index]
        return self

    def calculate_scores(self, feature):
        if self.w is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=float)
        return feature @ self.w + self.b

    def predict(self, test_data):
        test_data = np.asarray(test_data, dtype=float)
        if test_data.ndim == 1:
            test_data = test_data.reshape(1, -1)
        if test_data.ndim != 2 or test_data.shape[1] != self.w.shape[0]:
            raise ValueError("test_data 的特征数与训练数据不一致")

        scores = self.calculate_scores(test_data)
        return np.where(scores >= 0, self.classes_[0], self.classes_[1])
