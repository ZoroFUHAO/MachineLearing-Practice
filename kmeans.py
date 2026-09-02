import numpy as np

class kmeans:
    def __init__(self, k, max_iter=500, varepsilon=1e-5, seed=42):
        self.k = k
        self.max_iter = max_iter
        self.varepsilon = varepsilon
        self.rng = np.random.default_rng(seed)
        self.centroids = None
        self.clusters = None

    def distance(self, one_sample, feature):
        one_sample = one_sample.reshape(1,-1)
        return np.sum(np.power(one_sample - feature, 2), axis=1)

    def init_centroids(self, feature):
        choice = self.rng.choice(feature.shape[0], self.k, replace=False)
        return feature[choice]
    
    def _closest_centroids(self, sample, centroids):
        dist = self.distance(sample, centroids)
        return np.argmin(dist)

    def create_clusters(self, centroids, feature):
        clusters = [[] for _ in range(self.k)]

        for sample_index, sample in enumerate(feature):
            centroids_index = self._closest_centroids(sample, centroids)
            clusters[centroids_index].append(sample_index)

        return clusters
        
    def update_centroids(self, clusters, feature):
        centroids = np.zeros((self.k, feature.shape[1]))

        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                centroids[i] = np.mean(feature[cluster], axis=0)
            else:
                centroids[i] = feature[self.rng.choice(feature.shape[0])]

        return centroids

    def fit(self, feature):
        feature = np.asarray(feature, dtype=float)
        if feature.ndim != 2 or feature.shape[0] == 0:
            raise ValueError("feature 必须是非空二维数组")
        if not isinstance(self.k, (int, np.integer)):
            raise ValueError("k 必须是整数")
        if not 1 <= self.k <= feature.shape[0]:
            raise ValueError("k 必须满足 1 <= k <= 样本数")

        centroids = self.init_centroids(feature)
        for _ in range(self.max_iter):
            clusters = self.create_clusters(centroids, feature)

            old_centroids = centroids.copy()
            new_centroids = self.update_centroids(clusters, feature)
            centroids = new_centroids
            if np.all(np.abs(old_centroids - new_centroids) < self.varepsilon):
                break

        self.centroids = centroids
        self.clusters = self.create_clusters(self.centroids, feature)
        return self

    def predict(self, feature):
        if self.centroids is None:
            raise ValueError("请先调用 fit")

        feature = np.asarray(feature, dtype=float)
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        if feature.ndim != 2:
            raise ValueError("feature 必须是二维数组")

        labels = np.empty(feature.shape[0], dtype=int)

        for sample_index, sample in enumerate(feature):
            labels[sample_index] = self._closest_centroids(
                sample, self.centroids
            )

        return labels

"""
"""
import numpy as np

def cal_distance(x, feature):
    return np.sqrt(np.sum(np.square(x-feature), axis=1))

def compute_cluster(feature, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for i in range(feature.shape[0]):
        distances = cal_distance(feature[i], centroids)
        index = np.argmin(distances)
        clusters[index].append(i)
    return clusters

def fit_kmeans(feature, k, max_iter, varepsilon):
    feature = np.asarray(feature, dtype=float)
    n = feature.shape[0]

    centroids_index = np.random.choice(n, k, replace=False)
    centroids = feature[centroids_index].copy()

    for _ in range(max_iter):
        # 根据质心算出聚类
        clusters = compute_cluster(feature, centroids)

        new_centroids = centroids.copy()
        # 根据聚类算出新质心
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                new_centroids[i] = feature[np.random.choice(n)]
            else:
                new_centroids[i] = (np.mean(feature[cluster], axis=0))

        if np.max(np.abs(new_centroids - centroids)) < varepsilon:
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids

def predict_kmeans(X, centroids):
    predict = []
    for x in X:
        distances = cal_distance(x, centroids)
        predict.append(np.argmin(distances))
    return predict