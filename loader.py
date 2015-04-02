import numpy as np

def load_packets():
    np.random.seed(42)

    with open('kddcup.data_10_percent_corrected', 'r') as f:
        lines = f.readlines()

    X = []
    Y = []
    protocol_type_dict = {}
    service_dict = {}
    sf_dict = {}
    class_dict = {'normal':0}

    np.random.shuffle(lines)
    for i, line in enumerate(lines):
        x = line.strip(' \t\n\r.').split(',')

        if protocol_type_dict.get(x[1]) is None:
            protocol_type_dict[x[1]] = len(protocol_type_dict)

        if service_dict.get(x[2]) is None:
            service_dict[x[2]] = len(service_dict)

        if sf_dict.get(x[3]) is None:
            sf_dict[x[3]] = len(sf_dict)

        if class_dict.get(x[-1]) is None:
            class_dict[x[-1]] = len(class_dict)

        x[1] = protocol_type_dict[x[1]]
        x[2] = service_dict[x[2]]
        x[3] = sf_dict[x[3]]
        x[-1] = class_dict[x[-1]]
        for i, feature in enumerate(x):
            x[i] = float(x[i])

        X.append(x[:-1])
        Y.append(x[-1])

    return X, Y
