#!/usr/bin/env python
#-*- coding:utf-8 -*-

from scipy.stats import binom
from scipy.optimize import bisect

def uBinBound(m,k,delta=0.05):
    """
    Calculates the upper bound of the risk,
    using the binomial tail approach of Langford
    """
    if k == m: return 1.
    else:
        f = lambda x: binom.cdf( k, m, x ) - delta
        return bisect( f, float(k)/m, 1.0 )

def lBinBound(m,k,delta=0.05):
    """
    Calculates the lower bound of the risk,
    using the binomial tail approach of Langford
    """
    return 1 - uBinBound( m, m-k, delta )
