#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os.path
import sys
import json

def generate_result_table():
    predictors = ['hypersphere', 'one_class_svm', 'svc_biclass', 'multiclass_hypersphere', 'multiclass_one_class_svm', 'svc']
    datasets = ['packets', 'arrhythmia', 'breast-cancer-wisconsin', 'ecoli', 'glass', 'iris', 'liver']

    for dataset in datasets:
        for predictor in predictors:
            filename = 'result/' + predictor + '/' + dataset
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    result = json.load(f)
                    sys.stdout.write("%5.3f <= %5.3f <= %5.3f" % (result['risk_lower_bound']*100, result['risk']*100, result['risk_upper_bound']*100))
            sys.stdout.write('\t')
        print

generate_result_table()
