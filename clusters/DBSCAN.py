import numpy as np

class dbscan:
    def __init__(self, eps, Min_pts):
        self.eps = eps
        self.min_pts = Min_pts

    def find_neighbors(self, j, X, eps):
        N = []
        for p in range(X.shape[0]):
            dist = np.sqrt(np.sum(np.square(X[j]-X[p])))
            if dist <= eps:
                N.append(p)
        return N

    def fit(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        clusters = [-1] * n
        visited = [False] * n
        cluster_id = 0

        for i in range(n):
            if visited[i]: continue

            visited[i] = True
            neighbors = self.find_neighbors(i, X, self.eps)

            if len(neighbors) < self.min_pts:
                continue

            clusters[i] = cluster_id
            seeds = neighbors[:]
            index = 0

            while index < len(seeds):
                j = seeds[index]

                if not visited[j]:
                    visited[j] = True
                    clusters[j] = cluster_id

                    neighbor = self.find_neighbors(j, X, self.eps)
                    if len(neighbor) >= self.min_pts:
                        for x in neighbor:
                            if x not in seeds:
                                seeds.append(x)
                if clusters[j] == -1:
                    clusters[j] = cluster_id

                index += 1

            cluster_id += 1
        return clusters

"""
"""
import numpy as np

def get_neighbors(y, X, eps):
    neighbors = []

    for i, x in enumerate(X):
        if np.sqrt(np.sum((x - y) ** 2)) <= eps:
            neighbors.append(i)

    return neighbors

def Dbscan(X, eps, min_pts):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    visited = [False] * n
    clusters = [-1] * n       # -1 表示噪声
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = get_neighbors(X[i], X, eps)

        if len(neighbors) < min_pts:
            continue

        clusters[i] = cluster_id
        seeds = neighbors[:]

        while seeds:
            node = seeds.pop()

            if clusters[node] == -1:
                clusters[node] = cluster_id

            if visited[node]:
                continue

            visited[node] = True
            neighbors_node = get_neighbors(X[node], X, eps)

            if len(neighbors_node) >= min_pts:
                seeds.extend(neighbors_node)

        cluster_id += 1
    return np.array(clusters)
