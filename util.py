#!/usr/bin/env python2
# -*- coding: utf-8 -*-

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def risk(Y_predicted, Y_true):
    return sum(1 for y_predict, y_true in zip(Y_predicted, Y_true) if y_predict != y_true) / float(len(Y_true)) 
