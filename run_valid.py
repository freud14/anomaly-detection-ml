import numpy as np
from validate import Validator

def validate(name, X, Y, normal_class):
    anomaly = [x for x, y in zip(X, Y) if y != normal_class]
    y_anomaly = [y for y in Y if y != normal_class]
    normal = [x for x, y in zip(X, Y) if y == normal_class]
    y_normal = [y for y in Y if y == normal_class]

    nu_range = np.logspace(-3, 0, 10) * 0.5
    gamma_range = np.logspace(-5, 4, 10)
    validator = Validator(X, Y, normal, y_normal, anomaly, y_anomaly)
    print validator.one_class_svm(name, nu_range, gamma_range)
    print validator.hyperspherical_predictor(name, nu_range, gamma_range)
    print validator.svc_biclass(name, nu_range, gamma_range)
    print validator.multiclass_hyperspherical_predictor(name, nu_range, gamma_range)
    print validator.multiclass_one_class_svm(name, nu_range, gamma_range)
    print validator.svc(name, nu_range, gamma_range)
