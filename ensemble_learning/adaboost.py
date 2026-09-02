import numpy as np

class AdaBoost:
    '''
    input:n_estimators(int):迭代轮数
          learning_rate(float):弱分类器权重缩减系数
    '''
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels.reshape(-1)
        self.M, self.N = datasets.shape
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = np.ones(self.M)/self.M
        # G(x)系数 alpha
        self.alpha = []
    #********* Begin *********#
    def _G(self, features, labels, weights):
        best_error = float("inf")
        best_v = None
        best_direct = None

        values = np.sort(np.unique(features))
        if len(values) == 1:
            thresholds = values
        else:
            thresholds = (values[:-1] + values[1:]) / 2.0

        for v in thresholds:
            for direct in ["positive", "negative"]:
                preds = np.array([self.G(x, v, direct) for x in features])
                error = np.sum(weights[preds != labels])

                if error < best_error:
                    best_error = error
                    best_v = v
                    best_direct = direct

        return best_error, best_v, best_direct

    def _alpha(self, error):
        error = np.clip(error, 1e-10, 1 - 1e-10)
        return 0.5 * np.log((1 - error) / error)

    def _Z(self, weights, a, clf):
        z = 0.0
        feature_index, v, direct = clf
        for i in range(self.M):
            pred = self.G(self.X[i, feature_index], v, direct)
            z += weights[i] * np.exp(-a * self.Y[i] * pred)
        return z

    def _w(self, a, clf, Z):
        feature_index, v, direct = clf
        new_weights = np.zeros(self.M)

        for i in range(self.M):
            pred = self.G(self.X[i, feature_index], v, direct)
            new_weights[i] = self.weights[i] * np.exp(-a * self.Y[i] * pred) / Z

        self.weights = new_weights

    def G(self, x, v, direct):
        if direct == "positive":
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        self.init_args(X, y)

        for _ in range(self.clf_num):
            best_error = float("inf")
            best_clf = None

            for feature_index in range(self.N):
                features = self.X[:, feature_index]
                error, v, direct = self._G(features, self.Y, self.weights)

                if error < best_error:
                    best_error = error
                    best_clf = (feature_index, v, direct)

            if best_error >= 0.5:
                break

            a = self.learning_rate * self._alpha(best_error)
            self.alpha.append(a)
            self.clf_sets.append(best_clf)
            if best_error <= 1e-10:
                break

            Z = self._Z(self.weights, a, best_clf)
            self._w(a, best_clf, Z)
        return self

    def predict(self, data):
        data = np.asarray(data).reshape(-1, self.N)
        result = []
        for x in data:
            score = 0.0

            for a, clf in zip(self.alpha, self.clf_sets):
                feature_index, v, direct = clf
                score += a * self.G(x[feature_index], v, direct)
            result.append(1 if score >= 0 else -1)

        return result
