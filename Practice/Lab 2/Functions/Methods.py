from numpy import ndarray, sqrt, dot, matmul, mean, array, exp, arange, zeros, diag, linalg, append
from numpy.linalg import norm
from numpy.random import uniform
from typing import Callable, List
from sklearn.utils import shuffle
from math import inf
from tqdm import tqdm
from pandas import DataFrame
from itertools import product


# Функции ошибки

def mse(y_test: ndarray, y_predicted: ndarray) -> float:
    return ((y_test - y_predicted) ** 2).mean()


def rmse(y_test: ndarray, y_predicted: ndarray) -> float:
    return sqrt(mse(y_test, y_predicted))


def nrmse(y_test: ndarray, y_predicted: ndarray) -> float:
    return rmse(y_test, y_predicted) / mean(y_test)


def smape(y_test: ndarray, y_predicted: ndarray) -> float:
    n = y_test.size
    result = 0
    for i in range(n):
        result += abs(y_test[i] - y_predicted[i]) / (abs(y_test[i]) + abs(y_predicted[i]))
    return result * 100 / n


# --------------------

# Градиенты

def mse_gradient(w: ndarray,
                 x: List[ndarray],
                 y: ndarray) -> ndarray:
    elements = len(x)
    return 2 * dot(dot(x, w) - y, x) / elements


def rmse_gradient(w: ndarray,
                  x: List[ndarray],
                  y: ndarray) -> ndarray:
    return mse_gradient(w, x, y) / (2 * rmse(y_test=y, y_predicted=matmul(x, w)))


def nrmse_gradient(w: ndarray,
                   x: List[ndarray],
                   y: ndarray) -> ndarray:
    return rmse_gradient(w, x, y) / abs(mean(y))


def smape_gradient(w: ndarray,
                   x: List[ndarray],
                   y: ndarray) -> ndarray:
    answer = []
    for i in range(y.size):
        xw = dot(x[i], w.T)
        numerator = abs(xw - y[i])
        denominator = abs(xw) + abs(y[i])
        if xw > 0:
            if xw > y[i]:
                coefficient = denominator - numerator
            else:
                coefficient = -denominator - numerator
        else:
            if xw > y[i]:
                coefficient = denominator + numerator
            else:
                coefficient = -denominator + numerator
        answer.append(x[i] * coefficient / denominator ** 2)
    return mean(answer, axis=0)


def smape_gradient_wolfram(w: ndarray,
                           x: List[ndarray],
                           y: ndarray) -> ndarray:
    answer = []
    elements = y.size
    for i in range(elements):
        xw = dot(x[i], w)
        numerators = x[i] * ((xw - y[i]) * (abs(xw) + abs(y[i])) - ((xw * ((xw - y[i]) ** 2)) / abs(xw)))
        denominator = (abs(xw - y[i]) * ((abs(xw) + abs(y[i])) ** 2))
        if denominator == 0:
            answer.append(zeros((elements,)))
        else:
            answer.append(numerators / denominator)
    return mean(answer, axis=0)


def count_gradient(w: ndarray,
                   x: List[ndarray],
                   y: ndarray,
                   gradient: Callable[[ndarray,  # w
                                       List[ndarray],  # x
                                       ndarray],  # y
                                      ndarray],
                   tau: float,
                   regularisation_gradient: Callable[[float,
                                                      ndarray],
                                                     ndarray]) -> ndarray:
    vector = gradient(w, x, y)
    vector += regularisation_gradient(tau, w)
    vector = vector.reshape((-1,))
    return vector / norm(vector)


# --------------------

# Регуляризация

def ridge(tau: float, w: ndarray) -> float:
    return tau * norm(w)


def lasso(tau: float, w: ndarray) -> float:
    return tau * norm(w, ord=1)


def elastic(tau: float, w: ndarray) -> float:
    return tau * norm(w, ord=1) + norm(w)


# --------------------

# Градиенты регуляризации

def ridge_gradient(tau: float, w: ndarray) -> ndarray:
    return 2 * tau * w


def lasso_gradient(tau: float, w: ndarray) -> ndarray:
    return array([tau for _ in range(w.size)])


def elastic_gradient(tau: float, w: ndarray) -> ndarray:
    return array([tau for _ in range(w.size)]) + 2 * w


# --------------------

# Методы

def svd(arguments_training: List[ndarray],
        answers_training: List[float],
        arguments_test: List[ndarray],
        answers_test: ndarray) -> DataFrame:
    v, d, u = linalg.svd(arguments_training, full_matrices=False)
    taus = exp(arange(start=-17, stop=18))
    data_frame = DataFrame(columns=['Tau', 'NRMSE Mistake', 'SMAPE Mistake', 'Coefficients'])
    for tau in tqdm(taus):
        diagonal = diag(array([l / (l * l + tau) for l in d]))
        coefficients = u.T @ diagonal @ v.T @ answers_training
        answers_predicted = matmul(arguments_test, coefficients)
        nrmse_mistake = nrmse(array(answers_test), answers_predicted)
        smape_mistake = smape(array(answers_test), answers_predicted)
        data_frame.loc[len(data_frame) - 1] = [tau, nrmse_mistake, smape_mistake, coefficients]
    return data_frame


def gradient_decent(arguments_size: int,
                    epsilon: float,
                    iterations: int,
                    elements: int,
                    alpha: float,
                    tau: float,
                    shuffled_training_arguments: List[ndarray],
                    shuffled_training_answers: ndarray,
                    gradient: Callable[[ndarray,  # w
                                        List[ndarray],  # x
                                        ndarray],  # y
                                       ndarray],
                    mistake: Callable[[ndarray,  # y_test
                                       ndarray],  # y_predicted
                                      float],
                    regularisation: Callable[[float,  # tau
                                              ndarray],  # w
                                             float],
                    regularisation_gradient: Callable[[float,  # tau
                                                       ndarray],  # w
                                                      ndarray],
                    step_strategy: Callable[[int], float],
                    package: int,
                    change_if_bigger: bool = True) -> tuple[ndarray, ndarray, float]:
    w = uniform(low=-1 / (2 * arguments_size), high=1 / (2 * arguments_size), size=arguments_size)
    q = mistake(shuffled_training_answers, dot(shuffled_training_arguments, w))
    mistakes = []
    for i in range(min(iterations, elements - package)):
        step = step_strategy(i)
        training_arguments = [shuffled_training_arguments[j] for j in range(i, i + package)]
        training_answers = array([shuffled_training_answers[j] for j in range(i, i + package)])
        w_new = w - step * count_gradient(w=w,
                                          x=training_arguments,
                                          y=array(training_answers),
                                          gradient=gradient,
                                          tau=tau,
                                          regularisation_gradient=regularisation_gradient)
        mistake_count = mistake(training_answers, dot(training_arguments, w_new.T).reshape(-1, ))
        regularisation_count = regularisation(tau, w_new)
        q_new = (1 - alpha) * q + alpha * (mistake_count + regularisation_count)
        if abs(q_new - q) < epsilon and norm(w_new - w) < epsilon:
            break
        if not change_if_bigger and q_new < q:
            continue
        w = w_new
        q = q_new
        mistakes.append(q)
    return w, array(mistakes), q


# --------------------

epsilons = exp([-10, -3, -1])
alphas = arange(start=0, stop=1.1, step=0.2)
tau_variants = exp([-13, -7, -1, 1, 7, 13])
regularisations = {'ridge': (ridge, ridge_gradient), 'lasso': (lasso, lasso_gradient),
                   'elastic': (elastic, elastic_gradient)}
regularisations_names = ['ridge', 'lasso', 'elastic']
steps_strategies = {'linear': 1, 'square': 2, 'cube': 3}
steps_names = ['linear', 'square', 'cube']
mus = exp([-7, -1, 0, 1, 7])


def best_gd(mistake: Callable[[ndarray,  # y_test
                               ndarray],  # y_predicted
                              float],
            mistake_gradient: Callable[[ndarray,  # w
                                        List[ndarray],  # x
                                        ndarray],  # y
                                       ndarray],
            arguments_size: int,
            package_size: int,
            training_arguments: List[ndarray],
            training_answers: ndarray,
            test_arguments: List[ndarray],
            test_answers: ndarray):
    data_frame = DataFrame(columns=['Regularisation', 'Tau', 'Step strategy', 'Mu', 'Alpha', 'Epsilon',
                                    'Change if bigger', 'NRMSE Mistake', 'SMAPE Mistake', 'Final Q',
                                    'Mistakes', 'Result'])
    shuffled_training_arguments, shuffled_training_answers = shuffle(training_arguments, training_answers,
                                                                     random_state=0)
    inputs = product(regularisations_names, steps_names, mus, tau_variants, epsilons, alphas, [True, False])
    for regularisation_name, step, mu, tau, epsilon, alpha, change in tqdm(inputs):
        regularisation, regularisation_gradient = regularisations[regularisation_name]
        step_strategy = lambda i: mu / (i + 1) ** steps_strategies[step]
        w, mistakes, q = gradient_decent(arguments_size=arguments_size,
                                         epsilon=epsilon,
                                         iterations=500,
                                         elements=training_answers.size,
                                         alpha=alpha,
                                         tau=tau,
                                         shuffled_training_arguments=shuffled_training_arguments,
                                         shuffled_training_answers=shuffled_training_answers,
                                         gradient=mistake_gradient,
                                         mistake=mistake,
                                         regularisation=regularisation,
                                         regularisation_gradient=regularisation_gradient,
                                         step_strategy=step_strategy,
                                         package=package_size,
                                         change_if_bigger=change)
        predicted_answers = dot(test_arguments, w.T).reshape((-1,))
        final_nrmse = nrmse(test_answers, predicted_answers)
        final_smape = smape(test_answers, predicted_answers)
        data_frame.loc[len(data_frame) - 1] = [regularisation_name, tau, step, mu, alpha, epsilon, change,
                                               final_nrmse, final_smape, q, mistakes, w]
    return data_frame
