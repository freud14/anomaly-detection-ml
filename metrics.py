__author__ = 'prtos'
import numpy as np


def false_negative_rate(y_true, y_pred, label_positive_class=1):
    """

    :param y_true:
    :param y_pred:
    :param label_positive_class:
    :return:
    """
    n_false_negative = np.sum(1*(y_pred != label_positive_class) *
                             (1*(y_true == label_positive_class)))
    n_positifs = np.sum(y_true == label_positive_class)
    return (1.0 * n_false_negative) / n_positifs


def false_positive_rate(y_true, y_pred, label_negative_class=-1):
    """

    :param y_true:
    :param y_pred:
    :param label_positive_class:
    :return:
    """
    n_false_positifs = np.sum(1*(y_pred != label_negative_class) *
                              (1*(y_true == label_negative_class)))
    n_negatives = np.sum(y_true == label_negative_class)
    return (1.0 * n_false_positifs) / n_negatives


def np_score(y_true, y_pred, nu):
    """

    :param y_true:
    :param y_pred:
    :param nu:
    :return:
    """
    if nu != 0:
        score = max(0, false_positive_rate(y_true, y_pred) - nu) / nu + \
               false_negative_rate(y_true, y_pred)
    else:
        score  = -1

    return score
