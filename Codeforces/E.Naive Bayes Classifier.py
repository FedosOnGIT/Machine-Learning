from decimal import Decimal

import numpy as np


class Bayes:

    # number: int
    # lambdas: list[int]
    # alpha: int

    def __init__(self, number, lambdas, alpha):
        self.classes = []
        for i in range(number):
            self.classes.append({})
        self.number = number
        self.lambdas = lambdas
        self.alpha = alpha
        self.words = set()
        self.probabilities = np.zeros(number, dtype=Decimal)
        self.non_empty = np.full(shape=number, fill_value=False)

    # sentences: list[list[set[str]]]
    # number: int
    def fit(self, sentences, number):
        for i in range(self.number):
            for sentence in sentences[i]:
                self.words = self.words.union(sentence)
            if len(sentences[i]) != 0:
                self.probabilities[i] = Decimal(self.lambdas[i] * len(sentences[i]) / np.log(number))
                self.non_empty[i] = True
        for word in self.words:
            for i in range(self.number):
                present = 0
                for sentence in sentences[i]:
                    if word in sentence:
                        present += 1
                self.classes[i][word] = Decimal((present + self.alpha) / (len(sentences[i]) + 2 * self.alpha))
        for word in self.words:
            for i in range(self.number):
                if self.non_empty[i]:
                    self.probabilities[i] *= (Decimal(1) - self.classes[i][word])

    # sentence: set[str]

    # returns: np.ndarray
    def predict(self, sentence):
        probabilities = np.array(self.probabilities.copy())
        for word in sentence:
            if word in self.words:
                for i in range(self.number):
                    if self.non_empty[i]:
                        probabilities[i] /= Decimal(1) - self.classes[i][word]
                        probabilities[i] *= self.classes[i][word]
        denominator = np.sum(probabilities)
        return probabilities / denominator


classes_number = int(input())
bayes = Bayes(number=classes_number, lambdas=list(map(int, input().split())), alpha=int(input()))
train_size = int(input())
train_sentences = [[] for _ in range(classes_number)]
for i in range(train_size):
    sentence = list(map(str, input().split()))
    index = int(sentence[0]) - 1
    train_sentences[index].append(set(sentence[2:]))
bayes.fit(sentences=train_sentences, number=train_size)
test_size = int(input())
for i in range(test_size):
    sentence = list(map(str, input().split()))
    print(*bayes.predict(sentence=set(sentence[1:])))
