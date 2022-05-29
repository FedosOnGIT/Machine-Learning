import numpy as np


# w: np.ndarray
# x: np.ndarray
# y: float

def smape_gradient(w, x, y):
    xw = x @ w
    denominator = (np.abs(y) + np.abs(xw)) ** 2
    if y > xw:
        if xw > 0:
            numerator = -(np.abs(y) + np.abs(xw)) - np.abs(y - xw)
        else:
            numerator = -(np.abs(y) + np.abs(xw)) + np.abs(y - xw)
    elif y == xw:
        return np.zeros((w.size,))
    else:
        if xw > 0:
            numerator = (np.abs(y) + np.abs(xw)) - np.abs(y - xw)
        else:
            numerator = (np.abs(y) + np.abs(xw)) + np.abs(y - xw)
    return x * numerator / denominator


# def smape_gradient(w, x, y):
#     xw = x @ w
#     if y == xw:
#         return np.zeros(x.size)
#     numerator = (xw - y) * (np.abs(xw) + np.abs(y)) - (xw * (y - xw) ** 2) / np.abs(xw)
#     denominator = (np.abs(xw) + np.abs(y)) ** 2 * np.abs(y - xw)
#     return x * numerator / denominator

n, m = map(int, input().split())
features = []
answers = []
for i in range(n):
    values = list(map(float, input().split()))
    features.append(values[:m])
    features[i].append(1)
    answers.append(values[m])
weights = np.array(list(map(float, input().split())))
for i in range(n):
    gradient = smape_gradient(w=weights, x=np.array(features[i]), y=answers[i])
    print(*gradient)
