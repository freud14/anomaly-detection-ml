#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm, multiclass
from hyperspherical import HypersphericalPredictor
from util import *
from metrics import np_score
import json

class Validator:
    def __init__(self, X, Y, normal, y_normal, anomaly, y_anomaly):
        self.X = X
        self.Y = Y
        self.normal = normal
        self.anomaly = anomaly
        self.n_train = int(0.5 * len(self.normal))
        self.n_train_w_anomaly = int(0.25 * len(self.anomaly))
        self.n_valid_normal = int(0.25 * len(self.normal))
        self.n_valid_anomaly = int(0.25 * len(self.anomaly))
        self.n_test_normal = int(0.25 * len(self.normal))
        self.n_test_anomaly = int(0.50 * len(self.anomaly))

        train_normal = self.normal[:self.n_train]
        train_real_y_normal = y_normal[:self.n_train]
        train_anomaly = self.anomaly[:self.n_train_w_anomaly]
        train_real_y_anomaly = y_anomaly[:self.n_train_w_anomaly]
        self.train = train_normal
        self.train_w_anomaly = np.append(train_normal, train_anomaly, axis=0)
        self.y_train_w_anomaly = np.append(np.full(len(train_normal), 1), np.full(len(train_anomaly), -1))
        self.real_y_train_w_anomaly = np.append(train_real_y_normal, train_real_y_anomaly)

        valid_normal = self.normal[self.n_train:self.n_train + self.n_valid_normal]
        valid_real_y_normal = y_normal[self.n_train:self.n_train + self.n_valid_normal]
        valid_anomaly = self.anomaly[self.n_train_w_anomaly:self.n_train_w_anomaly + self.n_valid_anomaly]
        valid_real_y_anomaly = y_anomaly[self.n_train_w_anomaly:self.n_train_w_anomaly + self.n_valid_anomaly]
        self.valid = np.append(valid_normal, valid_anomaly, axis=0)
        self.y_valid = np.append(np.full(len(valid_normal), 1), np.full(len(valid_anomaly), -1))
        self.real_y_valid = np.append(valid_real_y_normal, valid_real_y_anomaly)

        test_normal = self.normal[self.n_train + self.n_valid_normal:]
        test_real_y_normal = y_normal[self.n_train + self.n_valid_normal:]
        test_anomaly = self.anomaly[self.n_train_w_anomaly + self.n_valid_anomaly:]
        test_real_y_anomaly = y_anomaly[self.n_train_w_anomaly + self.n_valid_anomaly:]
        self.test = np.append(test_normal, test_anomaly, axis=0)
        self.y_test = np.append(np.full(len(test_normal), 1), np.full(len(test_anomaly), -1))
        self.real_y_test = np.append(test_real_y_normal, test_real_y_anomaly)

    def hyperspherical_predictor(self, dataset, nu_range, gamma_range):
        return self._validate_anomaly_detector('result/hypersphere/' + dataset, HypersphericalPredictor, nu_range, gamma_range)

    def one_class_svm(self, dataset, nu_range, gamma_range):
        return self._validate_anomaly_detector('result/one_class_svm/' + dataset, svm.OneClassSVM, nu_range, gamma_range)

    def _validate_anomaly_detector(self, filename, predictor, nu_range, gamma_range):
        test = (-1, -1, -1, -1)

        best_risk_valid = -1
        best_nu = 0
        best_gamma = 0
        for nu in nu_range:
            best_gamma_np = 0
            best_np_valid = -1
            best_risk_np = -1
            for gamma in gamma_range:
                    estimator = predictor(nu=nu, kernel='rbf', gamma=gamma)
                    estimator.fit(self.train)

                    y_valid_predict = estimator.predict(self.valid)

                    np_valid = np_score(self.y_valid, y_valid_predict, nu)

                    risk_np = risk(y_valid_predict, self.y_valid)

                    if best_np_valid == -1 or np_valid < best_np_valid:
                        best_np_valid = np_valid
                        best_gamma_np = gamma
                        best_risk_np = risk_np
                    print "{}\t{}\t{}\t{}".format(nu, gamma, np_valid, risk_np)

                    if test[2] == -1 or np_valid < test[2]:
                        test = (nu, gamma, np_valid, risk_np)

            print (nu, best_gamma_np, best_np_valid, best_risk_np)

            if best_risk_valid == -1 or best_risk_np < best_risk_valid:
                best_risk_valid = best_risk_np
                best_nu = nu
                best_gamma = best_gamma_np
        print test

        estimator = predictor(nu=best_nu, kernel='rbf', gamma=best_gamma)
        estimator.fit(self.train_w_anomaly, self.y_train_w_anomaly)
        y_test_predict = estimator.predict(self.test)
        risk_test = risk(y_test_predict, self.y_test)
        risk_bound_test = risk_upper_bound(y_test_predict, self.y_test)

        print "%5.3f <= %5.3f <= %5.3f" % (risk_bound_test[0]*100, risk_test*100, risk_bound_test[1]*100)

        self._generate_result_file(filename, best_nu, best_gamma, risk_test, risk_bound_test)

        return best_nu, best_gamma, best_risk_valid, risk_test, risk_bound_test

    def svc_biclass(self, dataset, nu_range, gamma_range):
        best_risk_valid = -1
        best_nu = 0
        best_gamma = 0
        for nu in nu_range:
            for gamma in gamma_range:
                svc = svm.SVC(C=1./(nu*len(self.train_w_anomaly)), kernel='rbf', gamma=gamma)
                svc.fit(self.train_w_anomaly, self.y_train_w_anomaly)

                y_valid_predict = svc.predict(self.valid)
                risk_valid = risk(y_valid_predict, self.y_valid)

                if best_risk_valid == -1 or risk_valid < best_risk_valid:
                    best_risk_valid = risk_valid
                    best_nu = nu
                    best_gamma = gamma
                print "{}\t{}\t{}".format(nu, gamma, risk_valid)

        svc = svm.SVC(C=1./(best_nu*len(self.train_w_anomaly)), kernel='rbf', gamma=best_gamma)
        svc.fit(self.train_w_anomaly, self.y_train_w_anomaly)
        y_test_predict = svc.predict(self.test)
        risk_test = risk(y_test_predict, self.y_test)
        risk_bound_test = risk_upper_bound(y_test_predict, self.y_test)

        print "%5.3f <= %5.3f <= %5.3f" % (risk_bound_test[0]*100, risk_test*100, risk_bound_test[1]*100)

        self._generate_result_file('result/svc_biclass/' + dataset, best_nu, best_gamma, risk_test, risk_bound_test)

        return best_nu, best_gamma, best_risk_valid, risk_test, risk_bound_test

    def multiclass_hyperspherical_predictor(self, dataset, nu_range, gamma_range):
        return self._validate_multiclass('result/multiclass_hypersphere/' + dataset, HypersphericalPredictor, nu_range, gamma_range)

    def multiclass_one_class_svm(self, dataset, nu_range, gamma_range):
        return self._validate_multiclass('result/multiclass_one_class_svm/' + dataset, svm.OneClassSVM, nu_range, gamma_range)

    def _validate_multiclass(self, filename, predictor, nu_range, gamma_range):
        best_risk_valid = -1
        best_nu = 0
        best_gamma = 0
        for nu in nu_range:
            for gamma in gamma_range:
                multiclass_estimator = multiclass.OneVsRestClassifier(predictor(nu=nu, kernel='rbf', gamma=gamma))
                multiclass_estimator.fit(self.train_w_anomaly, self.real_y_train_w_anomaly)

                y_valid_predict = multiclass_estimator.predict(self.valid)
                risk_valid = risk(y_valid_predict, self.real_y_valid)

                if best_risk_valid == -1 or risk_valid < best_risk_valid:
                    best_risk_valid = risk_valid
                    best_nu = nu
                    best_gamma = gamma
                print "{}\t{}\t{}".format(nu, gamma, risk_valid)

        multiclass_estimator = multiclass.OneVsRestClassifier(predictor(nu=best_nu, kernel='rbf', gamma=best_gamma))
        multiclass_estimator.fit(self.train_w_anomaly, self.real_y_train_w_anomaly)
        y_test_predict = multiclass_estimator.predict(self.test)
        risk_test = risk(y_test_predict, self.real_y_test)
        risk_bound_test = risk_upper_bound(y_test_predict, self.real_y_test)

        print "%5.3f <= %5.3f <= %5.3f" % (risk_bound_test[0]*100, risk_test*100, risk_bound_test[1]*100)

        self._generate_result_file(filename, best_nu, best_gamma, risk_test, risk_bound_test)

        return best_nu, best_gamma, best_risk_valid, risk_test, risk_bound_test

    def svc(self, dataset, nu_range, gamma_range):
        best_risk_valid = -1
        best_nu = 0
        best_gamma = 0
        for nu in nu_range:
            for gamma in gamma_range:
                svc = svm.SVC(C=1./(nu*len(self.train_w_anomaly)), kernel='rbf', gamma=gamma)
                svc.fit(self.train_w_anomaly, self.real_y_train_w_anomaly)

                y_valid_predict = svc.predict(self.valid)
                risk_valid = risk(y_valid_predict, self.real_y_valid)

                if best_risk_valid == -1 or risk_valid < best_risk_valid:
                    best_risk_valid = risk_valid
                    best_nu = nu
                    best_gamma = gamma
                print "{}\t{}\t{}".format(nu, gamma, risk_valid)

        svc = svm.SVC(C=1./(best_nu*len(self.train_w_anomaly)), kernel='rbf', gamma=best_gamma)
        svc.fit(self.train_w_anomaly, self.real_y_train_w_anomaly)
        y_test_predict = svc.predict(self.test)
        risk_test = risk(y_test_predict, self.real_y_test)
        risk_bound_test = risk_upper_bound(y_test_predict, self.real_y_test)

        print "%5.3f <= %5.3f <= %5.3f" % (risk_bound_test[0]*100, risk_test*100, risk_bound_test[1]*100)

        self._generate_result_file('result/svc/' + dataset, best_nu, best_gamma, risk_test, risk_bound_test)

        return best_nu, best_gamma, best_risk_valid, risk_test, risk_bound_test

    def _generate_result_file(self, filename, best_nu, best_gamma, risk_test, risk_bound_test):
        with open(filename, 'w') as f:
            result = json.dumps({'risk': risk_test, 'risk_lower_bound': risk_bound_test[0], 'risk_upper_bound': risk_bound_test[1], 'nu': best_nu, 'gamma': best_gamma})
            f.write(result)
