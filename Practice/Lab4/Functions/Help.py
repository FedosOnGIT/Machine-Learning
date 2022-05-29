import os


def read_part(number: int) -> tuple[list[list[int]],
                                    list[list[int]]]:
    path = f'Messages/part{number}'
    files = os.listdir(path)
    legit = []
    spam = []
    for file_path in files:
        lines = open(f'{path}/{file_path}', 'r').readlines()
        subject = list(map(int, lines[0].split()[1:]))
        text = list(map(int, lines[2].split()))
        words = subject + text
        if 'legit' in file_path:
            legit.append(words)
        else:
            spam.append(words)
    return legit, spam
