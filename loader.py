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

    splitted_lines = [re.split(regexSeparator, line.strip(' \t\n\r.')) for line in lines]

    text_features = create_text_features_dict(splitted_lines)
    unknownIndexSet = get_unknown_feature_index_set(splitted_lines)

    np.random.shuffle(splitted_lines)
    for i, line in enumerate(splitted_lines):
        if '' in line:
            continue

        new_line = []
        for j, feature in enumerate(line):
            if j in unknownIndexSet:
                continue

            if j in text_features:
                if text_features[j].get(line[j]) is None:
                    text_features[j][line[j]] = len(text_features[j])
                line[j] = text_features[j][line[j]]

            new_line.append(float(line[j]))

        X.append(new_line[:-1])
        Y.append(new_line[-1])

    class_dict = get_class_dict(splitted_lines, text_features)
    return X, Y, text_features, class_dict

def get_unknown_feature_index_set(splitted_lines):
    unknownIndexSet = set()
    for i, line in enumerate(splitted_lines):
        for j, feature in enumerate(line):
            if feature == '?':
                unknownIndexSet.add(j)
    return unknownIndexSet

def create_text_features_dict(splitted_lines):
    text_features = {}
    i = 0
    first_example = splitted_lines[i]
    while '?' in first_example:
        i += 1
        first_example = splitted_lines[i]

    for j, feature in enumerate(first_example):
        try:
            float(feature)
        except ValueError:
            text_features[j] = {}

    return text_features

def get_class_dict(splitted_lines, text_features):
    class_dict = None
    n = len(splitted_lines[0])
    if n - 1 in text_features:
        class_dict = text_features[n - 1]
        del text_features[n - 1]
    return class_dict
