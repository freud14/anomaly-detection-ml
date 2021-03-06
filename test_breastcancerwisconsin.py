#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from loader import load_breastcancerwisconsin
from test import Tester
from run_valid import validate

nbTrain = 250
#nbTest = 2000


print "Beginning of loading."
X, Y, text_features, class_dict = load_breastcancerwisconsin();
normal_class = 2

nbTest = len(X) - nbTrain

X = preprocessing.normalize(X)
anomaly = [x for x, y in zip(X, Y) if y != normal_class]
normal = [x for x, y in zip(X, Y) if y == normal_class]
print "End of loading."


"""
test = Tester(X, Y, normal, anomaly, nbTrain, nbTest)
test.hyperspherical_predictor()
test.one_class_svm()
test.multiclass_hyperspherical_predictor()
test.multiclass_one_class_svm()
test.svc()
"""

validate('breast-cancer-wisconsin', X, Y, normal_class)
