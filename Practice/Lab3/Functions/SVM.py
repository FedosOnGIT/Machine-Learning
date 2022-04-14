import numpy as np
from Cores import Core
from scipy.optimize import minimize, Bounds
from typing import List


class SVM:

    def __init__(self, c: float):
        self.c = c
        self.w0 = 0
        self.w = None
        self.lambdas = None
        self.core = None
        self.x = None
        self.y = None
        self.size = None

    def l_function(self, lambdas: np.ndarray) -> float:
        second = 0
        for i in range(self.size):
            for j in range(self.size):
                second += lambdas[i] * lambdas[j] * self.y[i] * self.y[j] * self.core.matrix[i][j]
        return second / 2 - lambdas.sum()

    def l_jacobian(self, lambdas: np.ndarray) -> np.ndarray:
        jacobian = []
        for i in range(self.size):
            derivative = 0
            for j in range(self.size):
                plus = lambdas[j] * self.y[i] * self.y[j] * self.core.matrix[i][j]
                derivative += plus
            jacobian.append(derivative - 1)
        return np.array(jacobian)

    def fit(self, core: Core, x: List[np.ndarray], y: np.ndarray):
        self.core = core
        self.x = x
        self.y = y
        self.size = y.size

        bounds = Bounds(lb=np.zeros((self.size,)), ub=np.repeat(self.c, self.size))
        cs = np.repeat(self.c, self.size)
        constraints = {
            'type': 'eq',
            'fun': lambda lambdas: self.y.dot(lambdas),
            'jac': lambda lambdas: self.y
        }
        self.lambdas = minimize(fun=self.l_function, x0=cs, method='SLSQP', jac=self.l_jacobian,
                                constraints=constraints, bounds=bounds, options={'maxiter': 200,
                                                                                 'disp': False}).x
        self.w = (self.lambdas * self.y) @ self.x
        w0s = []
        epsilon = 0.001
        for i in range(self.size):
            if self.lambdas[i] + epsilon < self.c and self.lambdas[i] - epsilon > 0:
                w0s.append(self.w @ self.x[i] - self.y[i])
        if len(w0s) == 0:
            self.w0 = 0
        else:
            self.w0 = np.median(w0s)
        return self

    def predict(self, x: np.ndarray) -> int:
        prediction = 0
        for i in range(self.size):
            prediction += self.lambdas[i] * self.y[i] * self.core.core_function(x, self.x[i])
        return np.sign(prediction - self.w0)
