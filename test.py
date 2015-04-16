#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets, preprocessing, svm, multiclass
from hyperspherical import HypersphericalPredictor
from loader import load_packets
from util import *

nbTrain = 1000
nbTest = 2000


print "Beginning of loading."
X, Y, text_features, class_dict = load_packets();

#nbTest = len(X) - nbTrain

X = preprocessing.normalize(X)
anomaly = [x for x, y in zip(X, Y) if y != class_dict['normal']]
normal = [x for x, y in zip(X, Y) if y == class_dict['normal']]

train = normal[:nbTrain]
test_normal = normal[nbTrain:nbTrain + nbTest]
test_anormal = anomaly[:nbTest]

print "End of loading."



print "Beginning of testing of the hyperspherical predictor..."
# Test du prédicteur qui fonctionne avec une hypersphère
hypersphere = HypersphericalPredictor(kernel='rbf', gamma=0.5)
hypersphere.fit(train)

test_normal_predict = hypersphere.predict(test_normal)
print risk(test_normal_predict, np.full(len(test_normal_predict), 1))

test_anormal_predict = hypersphere.predict(test_anormal)
print risk(test_anormal_predict, np.full(len(test_anormal_predict), -1))
print "End of testing of the hyperspherical predictor."

"""
print "Beginning of testing of the one-class SVM..."
# Test du one-class SVM
one_class_svm = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.5)
one_class_svm.fit(train)

test_normal_predict = one_class_svm.predict(test_normal)
print risk(test_normal_predict, np.full(len(test_normal_predict), 1))

test_anormal_predict = one_class_svm.predict(test_anormal)
print risk(test_anormal_predict, np.full(len(test_anormal_predict), -1))
print "End of testing of the one-class SVM."
"""
"""
print "Beginning of testing of the multiclass hyperspherical predictor..."
# Test du prédicteur d'hypersphère en le transformant en prédicteur multi-classe.
multiclass_hypersphere = multiclass.OneVsRestClassifier(HypersphericalPredictor(kernel='rbf', gamma=0.5))
multiclass_hypersphere.fit(X[:nbTrain], Y[:nbTrain])

test_predict = multiclass_hypersphere.predict(X[nbTrain:nbTrain + nbTest])
print risk(test_predict, Y[nbTrain:nbTrain + nbTest])
print "End of testing of the multiclass hyperspherical predictor..."
"""
"""
print "Beginning of testing of the multiclass one-class SVM predictor..."
# Test du one-class SVM en le transformant en prédicteur multi-classe.
multiclass_hypersphere = multiclass.OneVsRestClassifier(svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.5))
multiclass_hypersphere.fit(X[:nbTrain], Y[:nbTrain])

test_predict = multiclass_hypersphere.predict(X[nbTrain:nbTrain + nbTest])
print risk(test_predict, Y[nbTrain:nbTrain + nbTest])
print "End of testing of the multiclass hyperspherical predictor..."
"""
"""
svc = svm.SVC(kernel='rbf', gamma=0.5)
svc.fit(X[:nbTrain], Y[:nbTrain])

test_predict = svc.predict(X[nbTrain:nbTrain + nbTest])
print risk(test_predict, Y[nbTrain:nbTrain + nbTest])
"""
