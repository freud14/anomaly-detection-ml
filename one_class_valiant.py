__author__ = 'prtos'
import numpy as np


# class Stumps:
#     def __init__(self, n_iter=10):
#         self.seuil_inf = 0
#         self.seuil_sup = 0
#         self.n_iter = n_iter
#
#     def train(self, x, y):
#         min = np.min(x)
#         step_size = (np.max(x) - min)/float(self.n_iter)
#         best_acc_inf, best_acc_sup = 0, 0
#         for i in range(-1, self.n_iter):
#             threshold = min + (i*step_size)
#             acc_inf = np.sum(1*(threshold <= x) == y)
#             acc_sup = np.sum(1*(x < threshold) == y)
#             if acc_inf > best_acc_inf:
#                 self.seuil_inf = threshold
#                 best_acc_inf = acc_inf
#             if acc_sup > best_acc_sup:
#                 self.seuil_sup = threshold
#                 best_acc_sup = acc_sup
#         return self
#
#     def predict(self, x):
#         preds = np.array(zip(1*(self.seuil_inf <= x), 1*(x < self.seuil_sup)))
#         return preds
#
# class DecisionStumps:
#     def __init__(self):
#         self.stumps = []
#
#     def train(self, x, y):
#         self.stumps = [Stumps(100).train(x[:, i], y) for i in range(x.shape[1])]
#         return self
#
#     def predict(self, x):
#         preds = tuple([stump.predict(x[:, i]) for i, stump in enumerate(self.stumps)])
#         preds = np.concatenate(preds, axis=1)
#         return preds
#
#
# class Valiant:
#     def __init__(self, percent_abnormal=0.01):
#         self.cv_index = []
#         self.ds = None
#         self.percent_abnormal = percent_abnormal
#
#     def train(self, x, y=None):
#         if y is None:
#             l = len(x)
#             la = int(l*self.percent_abnormal)
#             y = [0]*(l-la) + [1]*la
#             np.random.shuffle(y)
#
#         self.ds = DecisionStumps()
#         self.ds.train(x, y)
#         temp = self.ds.predict(x)
#         temp2 = 1 - temp
#         all_g = np.concatenate((temp, temp2), axis=1).T
#         print all_g.shape
#         positives = (y == 0)
#         for i, gi in enumerate(all_g):
#             if np.all(gi[positives]):
#                 self.cv_index += [i]
#         print len(self.cv_index)
#         return self
#
#     def predict(self, x):
#         temp = self.ds.predict(x)
#         temp2 = 1 - temp
#         all_g = np.concatenate((temp, temp2), axis=1).T
#         cv_g = all_g[self.cv_index].T
#         return np.all(cv_g, axis=0)
#

class StumpsOC:
    def __init__(self, p_unused=5):
        if not (0 <= p_unused <= 50):
            raise "p_unused must be between 0 and 50"
        self.seuil_inf = 0
        self.seuil_sup = 0
        self.p_unused = p_unused

    def train(self, x):
        self.seuil_inf = np.percentile(x, self.p_unused/2)
        self.seuil_sup = np.percentile(x, 100 - self.p_unused/2)
        return self

    def predict(self, x):
        preds = np.array((1*(self.seuil_inf < x))*(1*(x < self.seuil_sup)))
        return preds

class DecisionStumpsOC:
    def __init__(self, p_ununsed):
        self.stumps = []
        self.p_ununsed = p_ununsed

    def train(self, x):
        self.stumps = [StumpsOC(self.p_ununsed).train(x[:, i]) for i in range(x.shape[1])]
        return self

    def predict(self, x):
        preds = tuple([stump.predict(x[:, i]) for i, stump in enumerate(self.stumps)])
        preds = np.concatenate(preds, axis=1)
        return preds

class OneClassValiant:
    def __init__(self, percent_abnormal=0):
        self.cv_index = []
        self.ds = None
        self.percent_abnormal = percent_abnormal

    def train(self, x, y=None):
        # if y is None:
        #     l = len(x)
        #     la = int(l*self.percent_abnormal)
        #     y = [0]*(l-la) + [1]*la
        #     np.random.shuffle(y)

        self.ds = DecisionStumpsOC(self.percent_abnormal)
        self.ds.train(x)
        temp = self.ds.predict(x)
        print temp
        temp2 = 1 - temp
        all_g = np.concatenate((temp, temp2), axis=1).T
        print all_g.shape
        positives = (y == 0)
        for i, gi in enumerate(all_g):
            if np.all(gi[positives]):
                self.cv_index += [i]
        print len(self.cv_index)
        return self

    def predict(self, x):
        temp = self.ds.predict(x)
        temp2 = 1 - temp
        all_g = np.concatenate((temp, temp2), axis=1).T
        cv_g = all_g[self.cv_index].T
        return np.all(cv_g, axis=0)

