import numpy as np
from numpy.random import gamma
from sklearn.metrics import pairwise
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import resample
from sklearn import preprocessing

from numba import njit
from numba.typed import List

import uuid
import logging
import time

class SVM_RFE():
    def __init__(self, n_features_to_select, step=4):
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, X0, y, test=()):
        self.scores_ = {}
        self.test_scores_ = {}
        
        n_features_to_select = self.n_features_to_select
        n_features = X0.shape[1]
        if n_features_to_select is None:
            n_features_to_select = n_features
            
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)
    
        # np.sum(support_) is the number of selected features.
        # It starts at n_features and decreases every iteration.
        while np.sum(support_) > n_features_to_select:
            
            # Remaining features, represented with a list of indices.
            features = np.arange(n_features)[support_]
            X = X0[:, features]

            # Declare and train the SVM
            #with time_code('SVM #' + str(np.sum(support_))):
            estimator = LinearSVC(C=10, max_iter=5000, dual=False)
            estimator.fit(X, y)

            # Get importance and rank them
            importances = estimator.coef_ ** 2
            ranks = np.argsort(importances)

            # Flatten ranks, required for Multi-Class Classification.
            ranks = np.ravel(ranks)

            # Calculate t (step)
            threshold = min(self.step, np.sum(support_) - n_features_to_select)

            # Record score
            self.scores_[np.sum(support_) - 1] = estimator.score(X,y)
            if test: self.test_scores_[np.sum(support_) - 1] = estimator.score(test[0][:, features], test[1])

            # Eliminate the worse feature
            for i in range(0, threshold):
                selected_feature = features[ranks[i]]
                support_[selected_feature] = False
                ranking_[np.logical_not(support_)] += 1


        # Set final attributes
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

@njit
def combolutionMatrixCachePoly(X):
    m = X.shape[0]
    n = X.shape[0]
    C = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            C[i, j] = np.dot(X[i], X[j])
    return C

@njit
def combolutionMatrixCacheGauss(X):
    m = X.shape[0]
    n = X.shape[0]
    C = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            c = X[i] - X[j]
            C[i, j] = np.dot(c, c)
    return C

@njit
def updateMatrixPoly(X, K, C, Y, t, support, degree, gamma, coef0):
    for i in support:
        for j in support:
            K[i, j] = (((C[i,j] - X[i,t]*X[j,t]) * gamma + coef0) ** degree) * Y[i, j]

@njit
def updateMatrixGauss(X, K, C, Y, t, support, gamma):
    for i in support:
        for j in support:
            c = X[i,t] - X[j,t]
            K[i, j] = np.exp(-gamma * (C[i,j] - c*c)) * Y[i, j]

class SVM_RFE_KERNEL():
    def __init__(self, C=1, n_features_to_select = 1, step=4, kernel='linear', gamma = 1.0, degree = 3):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree

        self.nY = None
        self._H_y = None
        self._H_C = None
        self._H_K = None

        id = uuid.uuid4()
        self.logger = logging.getLogger(str(id))
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('logs/svm_rfe_' + str(id) + '.log')
        fileformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s",datefmt="%H:%M:%S")
        handler.setFormatter(fileformat)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        self.logger.warning('START')

    def computeKernelMatrix(self, X, Y):
        if self.kernel == 'linear' or self.kernel == 'poly':
            degree = 1 if self.kernel == 'linear' else self.degree
            self._H_K = pairwise.polynomial_kernel(X, Y, coef0=self.C, degree=degree)
            return self._H_K
        if self.kernel == 'rbf':
            self._H_K = pairwise.rbf_kernel(X, Y, gamma=self.gamma)
            return self._H_K
    
    def updatedKernelMatrix(self, X, i, supports):
        if self.kernel == 'linear' or self.kernel == 'poly':
            degree = 1 if self.kernel == 'linear' else self.degree
            gamma = 1.0 / X.shape[1]
            updateMatrixPoly(X, self._H_K, self._H_C, self._H_y, i, supports, degree, gamma, self.C)
        if self.kernel == 'rbf':
            updateMatrixGauss(X, self._H_K, self._H_C, self._H_y, i, supports, self.gamma)

    def computeHessianMatrix(self, K):
        return np.multiply(self._H_y, K)

    def combolutionMatrixCache(self, X):
        if self.kernel == 'linear' or self.kernel == 'poly':
            return combolutionMatrixCachePoly(X)
        else:
            return combolutionMatrixCacheGauss(X)

    def fit(self, X0, y, test=()):
        self.scores_ = {}
        self.test_scores_ = {}
        
        self.nY = y.copy()
        self.nY[self.nY == 0] = -1
        self._H_y = np.multiply.outer(y, y)
        
        n_features_to_select = self.n_features_to_select
        n_features = X0.shape[1]
        if n_features_to_select is None:
            n_features_to_select = n_features
            
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)
    
        # np.sum(support_) is the number of selected features.
        # It starts at n_features and decreases every iteration.
        elapsed_time = 0
        while np.sum(support_) > n_features_to_select:
            self.logger.info("PROGRESS " + "{:.2%}".format(1.0 - np.sum(support_) / n_features))

            # Remaining features, represented with a list of indices.
            features = np.arange(n_features)[support_]
            X = X0[:, features]
            #X = preprocessing.scale(X)

            # Precompute Hessian Matrix
            K = self.computeKernelMatrix(X, X)
            H = self.computeHessianMatrix(K)
            self._H_C = self.combolutionMatrixCache(X)

            # Declare and train the SVM
            estimator = SVC(C=self.C, max_iter=5000, kernel='precomputed')
            estimator.fit(K, y)

            # Record score
            self.scores_[np.sum(support_) - 1] = estimator.score(K,y)

            start_time = time.time()
            # Get importance and rank them
            a = np.ones(H.shape[0])
            a[estimator.support_] = estimator.dual_coef_[0]

            self.logger.info(f"SUPPORTS: {estimator.support_.shape}, TOTAL: {X.shape}")

            aHa = np.dot(np.dot(a, H), a)
            importances = np.zeros(X.shape[1])
            flist = list(range(0, X.shape[1]))
            #H_i = np.empty((X.shape[0], X.shape[0]))
            for i in flist:
                self.updatedKernelMatrix(X, i, List(estimator.support_))

                #X_i = X[:, np.delete(flist, i)]
                #K_i = self.computeKernelMatrix(X_i, X_i)
                #H_i = self.computeHessianMatrix(K_i)

                aH_ia = np.dot(np.dot(a, K), a) # K is alias of self._H_K
                importances[i] = (1/2) * (aHa - aH_ia)
           
            elapsed_time += time.time() - start_time

            ranks = np.argsort(importances)

            # Flatten ranks, required for Multi-Class Classification.
            ranks = np.ravel(ranks)

            # Calculate t (step)
            threshold = min(self.step, np.sum(support_) - n_features_to_select)


            # Eliminate the worse feature
            for i in range(0, threshold):
                selected_feature = features[ranks[i]]
                support_[selected_feature] = False
                ranking_[np.logical_not(support_)] += 1


        # Set final attributes
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        self.logger.info(f"ELAPSED IN HESSIAN: {elapsed_time}")

        return self


