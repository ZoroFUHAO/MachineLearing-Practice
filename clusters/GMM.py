import numpy as np

class gmm:
    def __init__(self, n_components=5, max_iter=1000, eps=1e-6):
        self.n = n_components
        self.max_iter = max_iter
        self.eps = eps

    def log_gauss_pdf(self, feature, mu, sigma):
        """
        X:     shape (样本数, 特征数)
        mu:    shape (特征数,)
        sigma: shape (特征数, 特征数)
        返回:  shape (样本数,)
        """
        d = feature.shape[1]
        diff = feature - mu

        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ValueError("协方差矩阵必须是正定的")

        # 等价于 diff @ np.linalg.inv(sigma)，但不要显式求逆
        solved = np.linalg.solve(sigma, diff.T).T
        mahalanobis = np.sum(diff * solved, axis=1)

        return -0.5 * (
            d * np.log(2 * np.pi)
            + log_det
            + mahalanobis
        )

    def softmax(self, log_gamma):
        max_feature = np.max(log_gamma, axis=1, keepdims=True)
        log_gamma -= max_feature
        gamma = np.exp(log_gamma)
        sum_gamma = np.sum(gamma, axis=1, keepdims=True)
        return gamma / sum_gamma

    def fit(self, feature):
        feature = np.asarray(feature)
        row, col = feature.shape

        self.alpha = np.ones(self.n) / self.n
        choice = np.random.choice(row, self.n, replace=False)
        self.mu = feature[choice].copy()
        self.sigma = np.array([np.eye(col) for _ in range(self.n)])

        for _ in range(self.max_iter):
            # E步
            log_gamma = np.zeros((row, self.n))

            for i in range(self.n):
                log_gamma[:, i] += (np.log(self.alpha[i] + self.eps)
                    + self.log_gauss_pdf(feature, self.mu[i], self.sigma[i])
                )

            # 对数概率转为普通概率，并按每行归一化
            gamma = self.softmax(log_gamma)
            
            # M步
            Nk = np.sum(gamma, axis=0)         
            safe_Nk = np.maximum(Nk, self.eps)

            # 更新混合系数
            self.alpha = Nk / row
            # 更新均值，shape: (k, col)
            self.mu = (gamma.T @ feature) / safe_Nk[:, np.newaxis]
            # 更新协方差矩阵
            for j in range(self.n):
                diff = feature - self.mu[j]

                self.sigma[j] = (
                    (diff * gamma[:, j][:, np.newaxis]).T @ diff
                    / safe_Nk[j]
                )
                # 保证协方差矩阵可逆
                self.sigma[j] += self.eps * np.eye(col)
        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data, dtype=float)
        row = len(test_data)

        log_prob = np.zeros((row, self.n))

        for j in range(self.n):
            log_prob[:, j] = (
                np.log(self.alpha[j] + self.eps)
                + self.log_gauss_pdf(
                    test_data,
                    self.mu[j],
                    self.sigma[j],
                )
            )

        # 返回每个样本概率最大的高斯分量编号
        return np.argmax(log_prob, axis=1)

"""
"""
class GMM:
    def __init__(self, k, max_iter, eps):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps

    def log_gauss_pdf(self, feature, mu, sigma):
        d = feature.shape[1]
        diff = feature - mu

        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0: raise ValueError

        solved = np.linalg.solve(sigma, diff.T).T
        mandistance = np.sum(diff * solved, axis=1)

        return -0.5 * (d * np.log(2 * np.pi) + logdet + mandistance)

    def softmax(self, log_gamma):
        log_gamma_max = np.max(log_gamma, axis=1, keepdims=True)
        log_gamma -= log_gamma_max
        gamma = np.exp(log_gamma)
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        return gamma / gamma_sum


    def fit(self, train_data):
        train_data = np.asarray(train_data)
        row, col = train_data.shape
        # 初始化
        self.alpha = np.ones(self.k) / self.k
        choice = np.random.choice(row, self.k, replace=False)
        self.mu = train_data[choice].copy()
        self.sigma = np.array([np.eye(col) for _ in range(self.k)])

        for _ in range(self.max_iter):
            # E步
            log_gamma = np.zeros((row, self.k))

            for i in range(self.k):
                # 计算gamma_ji
                log_gamma[:, i] += np.log(self.alpha[i] + self.eps) + self.log_gauss_pdf(train_data, self.mu[i], self.sigma[i])
            
            gamma = self.softmax(log_gamma)

            # M步
            N_k = np.sum(gamma, axis=0)

            self.alpha = N_k / row
            self.mu = gamma.T @ train_data / N_k[:, np.newaxis]

            for j in range(self.k):
                diff = train_data - self.mu[j]

                # self.sigma[j] = (diff * gamma[:, j][:, np.newaxis]).T @ diff / N_k[j]
                self.sigma[j] = (diff.T * gamma[:, j]) @ diff / N_k[j]
                self.sigma[j] += np.eye(col) * self.eps
        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        n = test_data.shape[0]

        log_prob = np.zeros((n, self.k))
        for j in range(self.k):
            log_prob[:, j] += np.log(self.alpha[j] + self.eps) + self.log_gauss_pdf(test_data, self.mu[j], self.sigma[j])
        
        index = np.argmax(log_prob, axis=1)
        return index
