import numpy as np
from validate import Validator
import matplotlib.pyplot as plt

def validate(name, X, Y, normal_class):
    anomaly = [x for x, y in zip(X, Y) if y != normal_class]
    y_anomaly = [y for y in Y if y != normal_class]
    normal = [x for x, y in zip(X, Y) if y == normal_class]
    y_normal = [y for y in Y if y == normal_class]

    nu_range = np.logspace(-3, 0, 30) * 0.2
    gamma_range = np.logspace(-5, 4, 10)
    validator = Validator(X, Y, normal, y_normal, anomaly, y_anomaly)

    """
    hypersphere_valid = validator.hyperspherical_predictor(name, nu_range, gamma_range)
    print hypersphere_valid[:-3]
    one_class_svm_valid = validator.one_class_svm(name, nu_range, gamma_range)
    print one_class_svm_valid[:-3]

    plt.plot(hypersphere_valid[-3], hypersphere_valid[-2], 'b')
    plt.plot(hypersphere_valid[-3], hypersphere_valid[-1], 'b--')
    plt.plot(one_class_svm_valid[-3], one_class_svm_valid[-2], 'g')
    plt.plot(one_class_svm_valid[-3], one_class_svm_valid[-1], 'g--')
    #plt.show()
    plt.savefig('graphs/' + name + '.png')
    """
    print validator.svc_biclass(name, nu_range, gamma_range)
    print validator.multiclass_hyperspherical_predictor(name, nu_range, gamma_range)
    print validator.multiclass_one_class_svm(name, nu_range, gamma_range)
    print validator.svc(name, nu_range, gamma_range)
