import numpy as np

class agnes:
    def __init__(self, k):
        self.k = k
        self.feature = None
        self.clusters = None

    def average_distance(self, cluster1, cluster2):
        points1 = self.feature[cluster1]
        points2 = self.feature[cluster2]

        distances = np.linalg.norm(
            points1[:, np.newaxis, :] - points2[np.newaxis, :, :],
            axis=2
        )
        return np.mean(distances)

    def complete_linkage(self, cluster1, cluster2):
        dist = -np.inf
        points1 = self.feature[cluster1]
        points2 = self.feature[cluster2]
        for x in points1:
            for y in points2:
                dist = max(dist, np.sqrt(np.sum(np.square(x - y))))
        return dist

    def fit(self, feature):
        self.feature = feature
        self.clusters = [[i] for i in range(feature.shape[0])]

        while len(self.clusters) > self.k:
            min_distance = np.inf
            merged_i, merged_j = -1, -1

            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    distance = self.complete_linkage(i, j)

                    if distance < min_distance:
                        min_distance = distance
                        merged_i, merged_j = i, j

            self.clusters[merged_i].extend(self.clusters[merged_j])
            del self.clusters[merged_j]

        return [self.feature[cluster].tolist() for cluster in self.clusters]

"""
"""
import numpy as np

def avarage_distance(cluster1, cluster2, X):
    m, n = len(cluster1), len(cluster2)
    dist = 0
    for i in range(m):
        dist += np.sum(np.sqrt(np.sum(np.square(X[cluster1[i]] - X[cluster2]), axis=1)))
    return dist / (m * n)

def Agnes(X, k):
    X = np.asarray(X)
    n, m = X.shape
    clusters = [[i] for i in range(n)]

    while len(clusters) > k:
        min_dist = np.inf
        merged_i, merged_j = 0, 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = avarage_distance(clusters[i], clusters[j], X)
                if dist < min_dist:
                    min_dist = dist
                    merged_i, merged_j = i, j

        clusters[merged_i].extend(clusters[merged_j])
        del clusters[merged_j]

    return [X[indices] for indices in clusters]