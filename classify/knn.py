import numpy as np

class knn:
    def __init__(self, k, p):
        self.p = p
        self.k = k

    def fit(self, train_data, train_label):
        self.train_data = np.asarray(train_data)
        self.train_label = np.asarray(train_label).reshape(-1)
        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        result = []

        for x in test_data:
            distances = np.sum(
                np.power(np.abs(x - self.train_data), self.p), axis=1
            ) ** (1 / self.p)
            choices = np.argsort(distances)[:self.k]
            choice_label = self.train_label[choices]
            classes, counts = np.unique(choice_label, return_counts=True)
            indexes = np.flatnonzero(counts == counts.max())
            if len(indexes) == 1:
                result.append(classes[indexes[0]])
                continue
            neighbor_distances = distances[choices]

            dist = np.array([
                neighbor_distances[choice_label == label].sum()
                for label in classes[indexes]
            ])

            result.append(classes[indexes[np.argmin(dist)]])
        return result

"""

"""
def knn(test_feature, feature, label, k):
    X = np.asarray(feature)
    y = np.asarray(label).reshape(-1)
    test_feature = np.asarray(test_feature)

    def cal_distance(x, feature):
        return np.sqrt(np.sum(np.square(x - feature), axis=1))

    result = []
    for x in test_feature:
        distances = cal_distance(x, X)
        choice = np.argsort(distances)[:k]

        label_classes, label_counts = np.unique(label[choice], return_counts=True)
        indexes = np.flatnonzero(label_counts == label_counts.max())
        if len(indexes) == 1:
            result.append(label_classes[indexes[0]])
            continue

        neighbor_distances = distances[choice]
        neighbor_labels = label[choice]
        sum_distances = np.array([
            neighbor_distances[neighbor_labels == c].sum()
            for c in label_classes[indexes]
        ])
        min_distance_index = np.argmin(sum_distances)
        result.append(label_classes[indexes[min_distance_index]])
    return result
