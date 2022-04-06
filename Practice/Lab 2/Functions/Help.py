from typing import List, Tuple

from numpy import ndarray, array, append
from tqdm import tqdm

# Чтение файла

def create_sample(lines: List[str], plus: int, size: int) -> tuple[list[ndarray], ndarray]:
    parameters = []
    answers = []
    for i in tqdm(range(size)):
        row = list(map(float, lines[i + plus].split()))
        x_parameters = row[:-1]
        x_parameters.append(1)
        parameters.append(array(x_parameters))
        answers.append(row[-1])
    return parameters, array(answers)

def read_txt(path: str) -> tuple[int, int, int, tuple[list[ndarray], ndarray], tuple[list[ndarray], ndarray]]:
    f = open(path, 'r')
    lines = f.readlines()
    parameters_size = int(lines[0])
    training_size = int(lines[1])
    training_parameters, training_answers = create_sample(lines, 2, training_size)
    test_size = int(lines[training_size + 2])
    test_parameters, test_answers = create_sample(lines, training_size + 3, test_size)
    return (parameters_size + 1, training_size, test_size,
            (training_parameters, training_answers),
            (test_parameters, test_answers))