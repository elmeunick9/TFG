import numpy as np
from numpy.random import gamma
from sklearn.metrics import pairwise
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import resample
from sklearn import preprocessing

import uuid
import logging

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

class SVM_RFE_KERNEL():
    def __init__(self, C=1, n_features_to_select = 1, step=4, kernel='linear', gamma = 1.0):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.nY = None

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
        if self.kernel == 'linear': return pairwise.polynomial_kernel(X, Y, degree=1)
        if self.kernel == 'poly': return pairwise.polynomial_kernel(X, Y, coef0=self.C)
        if self.kernel == 'rbf': return pairwise.rbf_kernel(X, Y, gamma=self.gamma)

    def computeHessianMatrix(self, K, y):
        return np.multiply(np.multiply.outer(y, y), K)

    def fit(self, X0, y, test=()):
        self.scores_ = {}
        self.test_scores_ = {}
        
        self.nY = y.copy()
        self.nY[self.nY == 0] = -1
        
        n_features_to_select = self.n_features_to_select
        n_features = X0.shape[1]
        if n_features_to_select is None:
            n_features_to_select = n_features
            
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)
    
        # np.sum(support_) is the number of selected features.
        # It starts at n_features and decreases every iteration.
        while np.sum(support_) > n_features_to_select:
            self.logger.info("PROGRESS " + "{:.2%}".format(1.0 - np.sum(support_) / n_features))

            # Remaining features, represented with a list of indices.
            features = np.arange(n_features)[support_]
            X = X0[:, features]
            X = preprocessing.scale(X)

            # Precompute Hessian Matrix
            K = self.computeKernelMatrix(X, X)
            H = self.computeHessianMatrix(K, self.nY)

            # Declare and train the SVM
            estimator = SVC(C=self.C, max_iter=5000, kernel='precomputed')
            estimator.fit(K, y)

            # Get importance and rank them
            a = np.ones(H.shape[0])
            a[estimator.support_] = estimator.dual_coef_[0]
            aHa = np.dot(np.dot(a, H), a) 
            importances = np.zeros(X.shape[1])
            flist = list(range(0, X.shape[1]))
            for i in flist:
                X_i = X[:, np.delete(flist, i)]
                K_i = self.computeKernelMatrix(X_i, X_i)
                H_i = self.computeHessianMatrix(K_i, y)
                aH_ia = np.dot(np.dot(a, H_i), a) 
                importances[i] = (1/2) * (aHa - aH_ia)

            ranks = np.argsort(importances)

            # Flatten ranks, required for Multi-Class Classification.
            ranks = np.ravel(ranks)

            # Calculate t (step)
            threshold = min(self.step, np.sum(support_) - n_features_to_select)

            # Record score
            self.scores_[np.sum(support_) - 1] = estimator.score(K,y)

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


