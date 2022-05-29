from __future__ import annotations

import random
import sys

import numpy as np
import pandas as pd
import abc
from sklearn.model_selection import KFold


class Node:

    def __init__(self, children: np.ndarray, rule: tuple[str,  # feature name
                                                         int,  # 0 - linear, 1 - classification, 2 - default
                                                         np.ndarray,  # values
                                                         int]  # default
                 ):
        self.children = children
        self.rule = rule

    def predict(self, features: pd.Series) -> int:
        if self.rule[1] == 0:
            position = 0 if features[self.rule[0]] <= self.rule[2] else 1
            return self.children[position].predict(features)
        elif self.rule[1] == 1:
            position = np.where(self.rule[2] == features[self.rule[0]])
            if len(position) > 0:
                return self.children[np.take(position, 0)].predict(features)
            else:
                return self.rule[3]
        else:
            return self.rule[3]


class DecisionTree:

    def __init__(self, data: pd.DataFrame, weights: np.ndarray = None, depth: int = 5, enable_random: bool = False):
        self.data = data.copy()
        self.depth = depth
        if weights is None:
            self.data.insert(0, 'weights', np.ones(self.data.shape[0]))
        else:
            self.data.insert(0, 'weights', weights)

        self.enable_random = enable_random
        self.root = self.train(0, self.data.index.to_numpy())

    @abc.abstractmethod
    def functional(self, counts: np.ndarray) -> float:
        raise NotImplementedError

    def branch_criterion(self, all_weights: np.ndarray, counts: np.ndarray,
                         children: list[tuple[np.ndarray, np.ndarray]]) -> float:
        all_weight = np.sum(all_weights)
        return self.functional(counts) - np.sum(
            [np.sum(child[0]) / all_weight * self.functional(child[1]) for child in children])

    def train(self, current_depth: int, indices: np.ndarray) -> Node:
        work = self.data.loc[indices]
        classes = work['class'].value_counts().rename_axis('class').reset_index(name='counts')
        default_help = work.groupby('class', as_index=False).sum()
        default = default_help[default_help['weights'] == default_help['weights'].max()]['class'].iloc[0]
        if classes.shape[0] == 1:
            return Node(children=np.array([]), rule=(work.columns[0], 2, np.array([]), default))
        if current_depth > self.depth:
            return Node(children=np.array([]), rule=(work.columns[0], 2, np.array([]), default))
        features = work.columns.drop(['class', 'weights'])
        best_indices = []
        best_criterion = 0
        best_rule = (work.columns[0], 2, np.array([]), default)

        for feature in features:
            work = work.sort_values(by=feature)
            values = work[feature].unique()
            values = np.sort(values)
            for i in range(values.size - 1):
                left = work[work[feature] <= values[i]]
                right = work[work[feature] > values[i]]
                current_indices = [left.index.to_numpy(), right.index.to_numpy()]
                children_branch = [(left['weights'].to_numpy(), left['class'].value_counts().to_numpy()),
                                   (right['weights'].to_numpy(), right['class'].value_counts().to_numpy())]
                criterion = self.branch_criterion(all_weights=work['weights'].to_numpy(),
                                                  counts=classes['counts'].to_numpy(),
                                                  children=children_branch)
                if criterion > best_criterion:
                    best_criterion = criterion
                    best_indices = current_indices
                    best_rule = (feature, 0, np.array(values[i]), default)
        children = np.array([self.train(current_depth + 1, child) for child in best_indices])
        return Node(children, best_rule)

    def predict(self, series: pd.Series) -> int:
        return self.root.predict(series)


class EntropyDecisionTree(DecisionTree):

    def functional(self, counts: np.ndarray) -> float:
        probabilities = counts / np.sum(counts)
        entropy = 0
        for probability in probabilities:
            entropy -= probability * np.log(probability)
        return entropy


class GiniGainDecisionTree(DecisionTree):

    def functional(self, counts: np.ndarray) -> float:
        probabilities = counts / np.sum(counts)
        gini = 1
        for probability in probabilities:
            gini -= probability ** 2
        return gini


class RandomForest:

    def __init__(self, data: pd.DataFrame, model: str):
        tree_models = {'entropy': lambda data, depth: EntropyDecisionTree(data, depth=depth),
                       'gini': lambda data, depth: GiniGainDecisionTree(data, depth=depth)}
        k_folds = KFold(n_splits=101, shuffle=True)
        self.trees = []
        for train, _ in k_folds.split(data):
            self.trees.append(tree_models[model](data.iloc[train], sys.maxsize))

    def predict(self, series: pd.Series) -> int:
        variants = [tree.predict(series) for tree in self.trees]
        return np.bincount(variants).argmax()
