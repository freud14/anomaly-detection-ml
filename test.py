#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm, multiclass
from hyperspherical import HypersphericalPredictor
from util import *

class Tester:
    def __init__(self, X, Y, normal, anomaly, n_train, n_test):
        self.X = X
        self.Y = Y
        self.normal = normal
        self.anomaly = anomaly
        self.n_train = n_train
        self.n_test = n_test

    def hyperspherical_predictor(self, nu=0.02, gamma=0.5):
        train_normal = self.normal[:self.n_train]
        test_normal = self.normal[self.n_train:self.n_train + self.n_test]
        test_anormal = self.anomaly[:self.n_test]

        print "Beginning of testing of the hyperspherical predictor..."
        # Test du prédicteur qui fonctionne avec une hypersphère
        hypersphere = HypersphericalPredictor(nu=nu, kernel='rbf', gamma=gamma)
        hypersphere.fit(train_normal)

        test_normal_predict = hypersphere.predict(test_normal)
        print risk(test_normal_predict, np.full(len(test_normal_predict), 1))

        test_anormal_predict = hypersphere.predict(test_anormal)
        print risk(test_anormal_predict, np.full(len(test_anormal_predict), -1))
        print "End of testing of the hyperspherical predictor."

    def one_class_svm(self, nu=0.02, gamma=0.5):
        train_normal = self.normal[:self.n_train]
        test_normal = self.normal[self.n_train:self.n_train + self.n_test]
        test_anormal = self.anomaly[:self.n_test]

        print "Beginning of testing of the one-class SVM..."
        # Test du one-class SVM
        one_class_svm = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        one_class_svm.fit(train_normal)

        test_normal_predict = one_class_svm.predict(test_normal)
        print risk(test_normal_predict, np.full(len(test_normal_predict), 1))

        test_anormal_predict = one_class_svm.predict(test_anormal)
        print risk(test_anormal_predict, np.full(len(test_anormal_predict), -1))
        print "End of testing of the one-class SVM."

    def multiclass_hyperspherical_predictor(self, nu=0.02, gamma=0.5):
        print "Beginning of testing of the multiclass hyperspherical predictor..."
        # Test du prédicteur d'hypersphère en le transformant en prédicteur multi-classe.
        multiclass_hypersphere = multiclass.OneVsRestClassifier(HypersphericalPredictor(nu=nu, kernel='rbf', gamma=gamma))
        multiclass_hypersphere.fit(self.X[:self.n_train], self.Y[:self.n_train])

        test_predict = multiclass_hypersphere.predict(self.X[self.n_train:self.n_train + self.n_test])
        print risk(test_predict, self.Y[self.n_train:self.n_train + self.n_test])
        print "End of testing of the multiclass hyperspherical predictor..."

    def multiclass_one_class_svm(self, nu=0.02, gamma=0.5):
        print "Beginning of testing of the multiclass one-class SVM predictor..."
        # Test du one-class SVM en le transformant en prédicteur multi-classe.
        multiclass_oneclass = multiclass.OneVsRestClassifier(svm.OneClassSVM(nu=nu, kernel='rbf', gamma=gamma))
        multiclass_oneclass.fit(self.X[:self.n_train], self.Y[:self.n_train])

        test_predict = multiclass_oneclass.predict(self.X[self.n_train:self.n_train + self.n_test])
        print risk(test_predict, self.Y[self.n_train:self.n_train + self.n_test])
        print "End of testing of the multiclass hyperspherical predictor..."

    def svc(self, C=1, gamma=0.5):
        svc = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        svc.fit(self.X[:self.n_train], self.Y[:self.n_train])

        test_predict = svc.predict(self.X[self.n_train:self.n_train + self.n_test])
        print risk(test_predict, self.Y[self.n_train:self.n_train + self.n_test])
