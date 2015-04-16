#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm, multiclass
from hyperspherical import HypersphericalPredictor
from util import *

class Tester:
    def __init__(self, X, Y, normal, anomaly, nbTrain, nbTest):
        self.X = X
        self.Y = Y
        self.normal = normal
        self.anomaly = anomaly
        self.nbTrain = nbTrain
        self.nbTest = nbTest

    def hyperspherical_predictor(self, gamma_hypersphere=0.5):
        train_normal = self.normal[:self.nbTrain]
        test_normal = self.normal[self.nbTrain:self.nbTrain + self.nbTest]
        test_anormal = self.anomaly[:self.nbTest]

        print "Beginning of testing of the hyperspherical predictor..."
        # Test du prédicteur qui fonctionne avec une hypersphère
        hypersphere = HypersphericalPredictor(kernel='rbf', gamma=gamma_hypersphere)
        hypersphere.fit(train_normal)

        test_normal_predict = hypersphere.predict(test_normal)
        print risk(test_normal_predict, np.full(len(test_normal_predict), 1))

        test_anormal_predict = hypersphere.predict(test_anormal)
        print risk(test_anormal_predict, np.full(len(test_anormal_predict), -1))
        print "End of testing of the hyperspherical predictor."

    def one_class_svm(self, gamma_one_class=0.5):
        train_normal = self.normal[:self.nbTrain]
        test_normal = self.normal[self.nbTrain:self.nbTrain + self.nbTest]
        test_anormal = self.anomaly[:self.nbTest]

        print "Beginning of testing of the one-class SVM..."
        # Test du one-class SVM
        one_class_svm = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=gamma_one_class)
        one_class_svm.fit(train_normal)

        test_normal_predict = one_class_svm.predict(test_normal)
        print risk(test_normal_predict, np.full(len(test_normal_predict), 1))

        test_anormal_predict = one_class_svm.predict(test_anormal)
        print risk(test_anormal_predict, np.full(len(test_anormal_predict), -1))
        print "End of testing of the one-class SVM."

    def multiclass_hyperspherical_predictor(self, gamma_multiclass_hypersphere=0.5):
        print "Beginning of testing of the multiclass hyperspherical predictor..."
        # Test du prédicteur d'hypersphère en le transformant en prédicteur multi-classe.
        multiclass_hypersphere = multiclass.OneVsRestClassifier(HypersphericalPredictor(kernel='rbf', gamma=gamma_multiclass_hypersphere))
        multiclass_hypersphere.fit(self.X[:self.nbTrain], self.Y[:self.nbTrain])

        test_predict = multiclass_hypersphere.predict(self.X[self.nbTrain:self.nbTrain + self.nbTest])
        print risk(test_predict, self.Y[self.nbTrain:self.nbTrain + self.nbTest])
        print "End of testing of the multiclass hyperspherical predictor..."

    def multiclass_one_class_svm(self, gamma_multiclass_oneclass=0.5):
        print "Beginning of testing of the multiclass one-class SVM predictor..."
        # Test du one-class SVM en le transformant en prédicteur multi-classe.
        multiclass_oneclass = multiclass.OneVsRestClassifier(svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=gamma_multiclass_oneclass))
        multiclass_oneclass.fit(self.X[:self.nbTrain], self.Y[:self.nbTrain])

        test_predict = multiclass_oneclass.predict(self.X[self.nbTrain:self.nbTrain + self.nbTest])
        print risk(test_predict, self.Y[self.nbTrain:self.nbTrain + self.nbTest])
        print "End of testing of the multiclass hyperspherical predictor..."

    def svc(self, gamma_svc=0.5):
        svc = svm.SVC(kernel='rbf', gamma=gamma_svc)
        svc.fit(self.X[:self.nbTrain], self.Y[:self.nbTrain])

        test_predict = svc.predict(self.X[self.nbTrain:self.nbTrain + self.nbTest])
        print risk(test_predict, self.Y[self.nbTrain:self.nbTrain + self.nbTest])
