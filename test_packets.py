#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from loader import load_packets
from test import Tester
from validate import Validator
from run_valid import validate


nbTrain = 1000
nbTest = 2000


print "Beginning of loading."
X, Y, text_features, class_dict = load_packets();
X, Y = X[:8000], Y[:8000] # On ne prend pas tout le dataset
normal_class = class_dict['normal']

#nbTest = len(X) - nbTrain

X = preprocessing.normalize(X)
anomaly = [x for x, y in zip(X, Y) if y != normal_class]
#y_anomaly = [y for y in Y if y != normal_class]
normal = [x for x, y in zip(X, Y) if y == normal_class]
#y_normal = [y for y in Y if y == normal_class]
print "End of loading."


#nu_range = np.logspace(-3, 0, 10) * 0.5
#gamma_range = np.logspace(-5, 4, 10)
#validator = Validator(X, Y, normal, y_normal, anomaly, y_anomaly)
#print validator.one_class_svm('packets', nu_range, gamma_range)
#print validator.hyperspherical_predictor('packets', nu_range, gamma_range)
#print validator.svc_biclass('packets', nu_range, gamma_range)
#print validator.multiclass_hyperspherical_predictor('packets', nu_range, gamma_range)
#print validator.multiclass_one_class_svm('packets', nu_range, gamma_range)
#print validator.svc('packets', nu_range, gamma_range)

validate('packets', X, Y, normal_class)

#test = Tester(X, Y, normal, anomaly, nbTrain, nbTest)
#test.one_class_svm(nu=1, gamma=0.001)
#test.one_class_svm()
#test.multiclass_hyperspherical_predictor()
#test.multiclass_one_class_svm()
#test.svc()
