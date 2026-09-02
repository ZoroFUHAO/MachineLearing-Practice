from practice.linear_model.LogisticRegression import logisticRegression
import numpy as np

"""
用逻辑回归代替2分类器
"""

class OVO:
    def __init__(self):
        self.model = []

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)

        classes = np.unique(train_label)

        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                choice = (train_label == classes[i]) | (train_label == classes[j])
                X = train_data[choice]
                y = (train_label[choice] == classes[i]).astype(int)
                model = logisticRegression()
                model.fit(X, y)
                self.model.append((model, classes[i], classes[j]))

        return self

    def predict(self, test_data):
        predicts = np.array([np.where(model.predict(test_data)==1, c1, c2) for model, c1, c2 in self.model])
        result = []

        for j in range(predicts.shape[1]):
            classes, count = np.unique(predicts[:,j], return_counts=True)
            result.append(classes[np.argmax(count)])
        return result

class OVR:
    def __init__(self):
        self.model = []

    def fit(self, train_data, train_label):
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label).reshape(-1)

        self.real_label = np.unique(train_label)

        for c in self.real_label:
            new_label = np.where(train_label == c, 1, 0)
            model = logisticRegression()
            model.fit(train_data, new_label)
            self.model.append(model)

        return self

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        predicts = np.array([model.predict_proba(test_data) for model in self.model])
        index = np.argmax(predicts, axis=0)
        return self.real_label[index]
        