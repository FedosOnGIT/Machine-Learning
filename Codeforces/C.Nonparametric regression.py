import numpy as np
from math import e, pi


def absolute(value: float, function) -> float:
    if np.abs(value) < 1:
        return function(value)
    else:
        return 0


class NPRegression:
    distances = {'euclidean': lambda x, y: np.sqrt(np.sum((x - y) ** 2)),
                 'manhattan': lambda x, y: np.sum(np.abs(x - y)),
                 'chebyshev': lambda x, y: np.max(np.abs(x - y))}

    cores = {'uniform': lambda value: absolute(value=value, function=lambda argument: 1 / 2),
             'triangular': lambda value: absolute(value=value, function=lambda argument: 1 - np.abs(argument)),
             'epanechnikov': lambda value: absolute(value=value, function=lambda argument: 3 / 4 * (1 - argument ** 2)),
             'quartic': lambda value: absolute(value=value,
                                               function=lambda argument: 15 / 16 * (1 - argument ** 2) ** 2),
             'triweight': lambda value: absolute(value=value,
                                                 function=lambda argument: 35 / 32 * (1 - argument ** 2) ** 3),
             'tricube': lambda value: absolute(value=value,
                                               function=lambda argument: 70 / 81 * (1 - np.abs(argument) ** 3) ** 3),
             'gaussian': lambda argument: 1 / np.sqrt(2 * pi) * e ** (- 1 / 2 * argument ** 2),
             'cosine': lambda value: absolute(value=value,
                                              function=lambda argument: pi / 4 * np.cos(pi / 2 * argument)),
             'logistic': lambda argument: 1 / (e ** argument + 2 + e ** (-argument)),
             'sigmoid': lambda argument: 2 / pi * 1 / (e ** argument + e ** (-argument))}
    windows = {'fixed': lambda dists, h: h,
               'variable': lambda dists, neighbour: np.sort(dists)[neighbour]}

    # distance_name: str
    # core_name: str
    # window_type: str
    # window_value: int
    def __init__(self, distance_name, core_name, window_type, window_value):
        self.answers_train = None
        self.features_train = None
        self.distance = self.distances[distance_name]
        self.core = self.cores[core_name]
        self.window = self.windows[window_type]
        self.window_value = window_value

    # features_train: list[np.ndarray]
    # answers_train: np.ndarray
    def fit(self, features_train, answers_train):
        self.features_train = features_train
        self.answers_train = answers_train

    # features: np.ndarray
    def predict(self, features):
        dists = np.array([self.distance(features, train) for train in self.features_train])
        h = self.window(dists, self.window_value)
        numerator = 0
        denominator = 0
        if h == 0:
            same = []
            for index in range(dists.size):
                if dists[index] == 0:
                    same.append(self.answers_train[index])
            if len(same) > 0:
                return np.mean(same)
        else:
            for index in range(dists.size):
                weight = self.core(dists[index] / h) if h != 0 else 0
                numerator += weight * self.answers_train[index]
                denominator += weight
            if denominator != 0:
                return numerator / denominator
        return np.mean(self.answers_train)


n, m = map(int, input().split())
features = []
answers = []
for i in range(n):
    values = list(map(int, input().split()))
    features.append(np.array(values[:m]))
    answers.append(values[m])
test = np.array(list(map(int, input().split())))
distance = input()
core = input()
window = input()
window_value = int(input())
regression = NPRegression(distance_name=distance, core_name=core, window_type=window, window_value=window_value)
regression.fit(features_train=features, answers_train=np.array(answers))
print(regression.predict(features=test))
