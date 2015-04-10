#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt.blas import dot
import numpy as np
from util import fullprint # Pour imprimer un Numpy Array au complet
from sklearn import metrics, base

class HypersphericalPredictor(base.BaseEstimator):
    def __init__(self, C=1.0, kernel='rbf', degree=None, gamma=None, coef0=None):
        self.C = C
        self.kernel = kernel
        if callable(kernel):
            self.kernel_func = kernel
        else:
            self.kernel_func = metrics.pairwise.kernel_metrics()[kernel]

        # Paramètres du noyau
        self.kwargs = {}
        if gamma is not None:
            self.gamma = gamma
            self.kwargs['gamma'] = self.gamma

        if degree is not None:
            self.degree = degree
            self.kwargs['degree'] = self.degree

        if coef0 is not None:
            self.coef0 = coef0
            self.kwargs['coef0'] = self.coef0

    def fit(self, X, y=None):
        # Lorsque le classificateur est utilisé par une stratégie multiclasse
        # un contre tous par scikit-learn, on passe un paramètre y dont les 1 de
        # la liste correspondent à la classe testée. On fait donc le détecteur
        # d'anomalie sur les exemples étiquettés 1.
        if y is not None:
            self.train = np.array([x for x,y in zip(X, y) if y == 1])
        else:
            self.train = np.asarray(X)

        m = len(self.train)

        if m == 0:
            raise Exception('Training set is empty.')

        # Calcul de la matrice K et de la matrice comprenant seulement k(x_i, x_i) pour tout i.
        K = self.kernel_func(self.train, self.train, **self.kwargs)
        Kii = [K[i][i] for i in xrange(m)]
        K = matrix(K)
        Kii = matrix(Kii)

        P = -K
        q = Kii

        # Modélisation du problème de programmation quadratique selon ce que
        # le solveur quadratique veut.

        # 0 <= alpha <= C
        G = matrix(np.append(-np.eye(m), np.eye(m), axis=0))
        h = matrix(np.append(np.full(m, 0), np.full(m, self.C)))

        # \sum \alpha_i = 1
        A = matrix(1.0, (1, m))
        b = matrix([1.0])

        # On demande au solveur de résoudre le problème.
        # On met P et q négatif parce qu'il s'agit d'une minimisation.
        alpha = qp(-P, -q, G, h, A, b)['x']

        # Calcul de la norme L2 du centre c de l'hypersphère
        self.l2c = dot(alpha.T * K, alpha)
        # Calcul de la norme L2 du rayon r
        self.r2 = dot(alpha, Kii) - self.l2c

        self.alpha = np.array(alpha).flatten()

        """
        # On supprime les petites valeurs de alpha pour garder seulement celle qui sont significatives.
        self.alpha = np.array([a if a >= 10**-3 else 0 for a in self.alpha])
        nonzero_alpha = np.flatnonzero(self.alpha)

        if nonzero_alpha.shape[0] < self.alpha.shape[0]:
            self.train = self.train[nonzero_alpha]

            #self.alpha = self.alpha[nonzero_alpha]
            #self.alpha = self.alpha / sum(self.alpha) # \sum \alpha_i = 1

            return self.fit(self.train)
        """
        return self

    def predict(self, X):
        # I(||phi(x) - c||^2 < r^2)
        return np.sign(-self._get_square_root_difference(X))

    def decision_function(self, X):
        return -self._get_square_root_difference(X)

    def _get_square_root_difference(self, X):
        """
        Cette méthode donne la différence entre ||phi(x) - c|| et r.
        """
        x = np.array(X)

        # t1 est le vecteur de k(x_i, x_i) tel que i = 1,...,m où m = len(X)
        t1 = np.array([self.kernel_func(x_i,x_i, **self.kwargs)[0][0] for x_i in x])
        # t2 est la matrice que \sum alpha_i k(x_i, x) pour chaque x \in X et x_i \in train
        t2 =  - 2 * self.alpha.dot(self.kernel_func(self.train, x, **self.kwargs))

        # Voir Note de cours Chapitre 2 page 58
        # On veut avoir la différence sans le carré (décision intuitive sans argument théorique)
        return (t1 + t2 + self.l2c)**0.5 - self.r2**0.5

    def _set_kernel_parameter(self):
        print self.gamma
