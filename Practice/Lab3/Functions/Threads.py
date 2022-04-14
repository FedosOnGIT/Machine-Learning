import threading
from typing import Callable, List
from SVM import SVM
import numpy as np


class SVMThread(threading.Thread):

    def __init__(self, function: Callable[[List[np.ndarray], np.ndarray], tuple[float, SVM]],
                 features: List[np.ndarray], answers: np.ndarray):
        self.svm = None
        self.accuracy = None
        self.function = function
        self.features = features
        self.answers = answers
        super().__init__()

    def run(self) -> None:
        self.accuracy, self.svm = self.function(self.features, self.answers)
