import numpy as np


# int
# list[list[int]]
# int

# tuple[int, int, int, int]

def calculate_positive(positive,
                       matrix,
                       size):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for column in range(size):
        if column == positive:
            true_positive += matrix[positive][positive]
        else:
            false_positive += matrix[positive][column]
            false_negative += matrix[column][positive]
            true_negative += matrix[column][column]
    return true_positive, true_negative, false_positive, false_negative


# int
# int
# int

# tuple[float, float]

def precision_recall(true_positive,
                     false_positive,
                     false_negative):
    if true_positive == 0:
        return 0, 0
    return true_positive / (true_positive + false_positive), true_positive / (true_positive + false_negative)


# float
# float

# float

def f_score(precision,
            recall):
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


matrix_size = int(input())
classification_matrix = []
weights = []
for i in range(matrix_size):
    elements = list(map(int, input().split()))
    classification_matrix.append(elements)
    weights.append(np.sum(elements))
weights = np.divide(weights, np.sum(weights))

positives_negatives = []
precisions_recalls = []
f_scores = []
for i in range(matrix_size):
    positives_negatives.append(calculate_positive(i, matrix=classification_matrix, size=matrix_size))
    precisions_recalls.append(precision_recall(true_positive=positives_negatives[i][0],
                                               false_positive=positives_negatives[i][2],
                                               false_negative=positives_negatives[i][3]))
    f_scores.append(f_score(precision=precisions_recalls[i][0],
                            recall=precisions_recalls[i][1]))

micro_true_positive, micro_true_negative, micro_false_positive, micro_false_negative = np.sum(
    a=positives_negatives * weights[:, np.newaxis],
    axis=0)
micro_precision, micro_recall = precision_recall(true_positive=micro_true_positive,
                                                 false_positive=micro_false_positive,
                                                 false_negative=micro_false_negative)
micro_average_f_score = f_score(precision=micro_precision, recall=micro_recall)
print(micro_average_f_score)

macro_precision, macro_recall = np.sum(precisions_recalls * weights[:, np.newaxis], axis=0)
macro_average_f_score = f_score(precision=macro_precision, recall=macro_recall)
print(macro_average_f_score)

average_f_score = np.sum(f_scores * weights)
print(average_f_score)
