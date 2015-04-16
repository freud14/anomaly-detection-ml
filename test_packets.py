#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from loader import load_packets
from test import Tester

nbTrain = 1000
nbTest = 2000


print "Beginning of loading."
#X, Y, text_features, class_dict = load_arrhythmia();
#X, Y, text_features, class_dict = load_breastcancerwisconsin();
#X, Y, text_features, class_dict = load_ecoli();
#X, Y, text_features, class_dict = load_glass();
#X, Y, text_features, class_dict = load_iris();
#X, Y, text_features, class_dict = load_liver();

X, Y, text_features, class_dict = load_packets();

#nbTest = len(X) - nbTrain

X = preprocessing.normalize(X)
anomaly = [x for x, y in zip(X, Y) if y != class_dict['normal']]
normal = [x for x, y in zip(X, Y) if y == class_dict['normal']]
print "End of loading."

test = Tester(X, Y, normal, anomaly, nbTrain, nbTest)
test.hyperspherical_predictor()
#test.one_class_svm()
#test.multiclass_hyperspherical_predictor()
#test.multiclass_one_class_svm()
#test.svc()
