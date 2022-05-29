import numpy as np


class NGramSpliter:

    def __init__(self, n: int):
        self.n = n

    def create_n_gram(self, words: list[int]) -> set[tuple]:
        words_n_grams = set()
        for i in range(len(words)):
            j = i
            n_gram = []
            while j - i < self.n:
                if j < len(words):
                    n_gram.append(words[j])
                else:
                    n_gram.append(-1)
                j += 1
            words_n_grams.add(tuple(n_gram))
        return words_n_grams

    def split(self, messages: list[list[int]]) -> list[set[tuple]]:
        n_grams = []
        for message in messages:
            n_grams.append(self.create_n_gram(message))
        return n_grams


class Counter:

    def __init__(self, parts: list[tuple[list[set[tuple]],
                                         list[set[tuple]]]]):
        self.dictionary = dict()
        self.all = [0, 0]
        self.alpha = 1
        self.legit = 0
        self.spam = 0
        for part in parts:
            self.plus(texts=part[0], position=0)
            self.plus(texts=part[1], position=1)

    def plus(self, texts: list[set[tuple]], position: int) -> None:
        for text in texts:
            for n_gram in text:
                key = tuple(n_gram)
                if key not in self.dictionary:
                    self.dictionary[key] = [0, 0]
                self.dictionary[key][position] += 1
            self.all[position] += 1

    def set_parameters(self, alpha: float) -> None:
        self.alpha = alpha
        self.legit = np.log(self.all[0] / np.sum(self.all))
        self.spam = np.log(self.all[1] / np.sum(self.all))
        for key, value in self.dictionary.items():
            legit_value = (value[0] + self.alpha) / (self.all[0] + 2 * self.alpha)
            spam_value = (value[1] + self.alpha) / (self.all[1] + 2 * self.alpha)
            self.legit += np.log(1 - legit_value)
            self.spam += np.log(1 - spam_value)

    def predict(self, text: set[tuple]) -> tuple[float, float]:
        legit = self.legit
        spam = self.spam
        for word in text:
            if word in self.dictionary:
                value = self.dictionary[word]
                legit_value = (value[0] + self.alpha) / (self.all[0] + 2 * self.alpha)
                spam_value = (value[1] + self.alpha) / (self.all[1] + 2 * self.alpha)
                legit += np.log(legit_value) - np.log(1 - legit_value)
                spam += np.log(spam_value) - np.log(1 - spam_value)
        return legit, spam


def calculate(legit: float, spam: float) -> float:
    legit_probability = np.exp(legit)
    spam_probability = np.exp(spam)
    summary = legit_probability + spam_probability
    if summary == 0:
        summary = np.abs(legit + spam)
        legit_probability = summary + legit
    return legit_probability / summary


