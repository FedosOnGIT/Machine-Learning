import numpy as np
import abc
from typing import List


class Core:

    def __init__(self):
        self.matrix = None

    def fit(self, x: List[np.ndarray]):
        self.matrix = np.array(
            [np.array([self.core_function(x[i], x[j]) for j in range(len(x))]) for i in range(len(x))])
        return self

    @abc.abstractmethod
    def core_function(self, x_1: np.ndarray, x_2: np.ndarray) -> float:
        raise NotImplementedError


class LinearCore(Core):

    def core_function(self, x_1: np.ndarray, x_2: np.ndarray) -> float:
        return x_1 @ x_2


class PolynomialCore(Core):

    def __init__(self, degree: int, plus: int):
        self.degree = degree
        self.plus = plus
        super().__init__()

    def core_function(self, x_1: np.ndarray, x_2: np.ndarray) -> float:
        return (x_1 @ x_2 + self.plus) ** self.degree


class GaussianCore(Core):

    def __init__(self, beta: int):
        self.beta = beta
        super().__init__()

    def core_function(self, x_1: np.ndarray, x_2: np.ndarray) -> float:
        return np.exp(- self.beta * np.linalg.norm(x_1 - x_2))
