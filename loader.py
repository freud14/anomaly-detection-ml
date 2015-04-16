import numpy as np
import re

def load_packets():
    return load_csv_file('kddcup.data_10_percent_corrected')

def load_arrhythmia():
    return load_csv_file('data/arrhythmia.data')

def load_breastcancerwisconsin():
    return load_csv_file('data/breast-cancer-wisconsin.data')

def load_ecoli():
    return load_csv_file('data/ecoli.data', regexSeparator='\\s*')

def load_glass():
    return load_csv_file('data/glass.data')

def load_iris():
    return load_csv_file('data/iris.data')

def load_liver():
    return load_csv_file('data/liver.data')

def load_csv_file(filename, regexSeparator=','):
    np.random.seed(42)

    with open(filename, 'r') as f:
        lines = f.readlines()

    X = []
    Y = []

    text_features = {}
    i = 0
    first_example = re.split(regexSeparator, lines[i].strip(' \t\n\r.'))
    while '?' in first_example:
        i += 1
        first_example = re.split(regexSeparator, lines[i].strip(' \t\n\r.'))
    del i
    for j, feature in enumerate(first_example):
        try:
            float(feature)
        except ValueError:
            text_features[j] = {}
    del j

    np.random.shuffle(lines)
    for i, line in enumerate(lines):
        x = re.split(regexSeparator, line.strip(' \t\n\r.'))

        for j, feature in enumerate(x):
            if '?' in x:
                continue

            if j in text_features:
                if text_features[j].get(x[j]) is None:
                    text_features[j][x[j]] = len(text_features[j])
                x[j] = text_features[j][x[j]]

            x[j] = float(x[j])

        X.append(x[:-1])
        Y.append(x[-1])

    class_dict = None
    if len(first_example) - 1 in text_features:
        class_dict = text_features[len(first_example) - 1]
        del text_features[len(first_example) - 1]
    return X, Y, text_features, class_dict
