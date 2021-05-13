import time
import numpy as np
from sklearn.svm import LinearSVC
from multiprocessing import Pool
import random

from svm_rfe import SVM_RFE, SVM_RFE_SAMPLING


class DSMethods:
    def __init__(self, n_features, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_features = n_features

    def randomSelection(self, CVal=10):
        start_time = time.time()
        random_scores_train = {}
        random_scores_test = {}
        random_selection = random.sample(range(0, self.n_features), self.n_features)

        for i in range(1, self.n_features, int(self.n_features/50)):
            features = random_selection[:i]
        
            svm = LinearSVC(C=CVal, max_iter=5000, dual=False)
            svm.fit(self.X_train[:,features], self.y_train)

            random_scores_train[i] = svm.score(self.X_train[:,features], self.y_train)
            random_scores_test[i] = svm.score(self.X_test[:,features], self.y_test)

        return random_scores_train, random_scores_test, time.time() - start_time

    def _svm_rfe(self, rfe, XT, yT, Xt, yt):
        start_time = time.time()
        rfe.fit(XT, yT)
        elapsed_time = time.time() - start_time

        train_scores = {}
        test_scores = {}
        test_selection = np.argsort(rfe.ranking_)

        for i in range(1, self.n_features, int(self.n_features/50)):
            features = test_selection[:i]

            svm = LinearSVC(C=10, max_iter=5000, dual=False)
            svm.fit(XT[:,features], yT)

            train_scores[i] = svm.score(XT[:,features], yT)
            test_scores[i] = svm.score(Xt[:,features], yt)

        return train_scores, test_scores, rfe.scores_, elapsed_time

    def _svm_rfe_only(self, rfe, XT, yT, Xt, yt):
        start_time = time.time()
        rfe.fit(XT, yT, (Xt, yt))
        elapsed_time = time.time() - start_time

        return rfe.scores_, rfe.test_scores_, rfe.scores_, elapsed_time

    def svm_rfe(self, args):
        train_index, test_index, step = args
        XT, Xt = self.X_train[train_index], self.X_train[test_index]
        yT, yt = self.y_train[train_index], self.y_train[test_index]
        rfe = SVM_RFE(n_features_to_select=1, step=step)
        return self._svm_rfe(rfe, XT, yT, Xt, yt)

    def svm_rfe_dynamic_step(self, args):
        stop = 1
        train_index, test_index, percentage, stop = args
        XT, Xt = self.X_train[train_index], self.X_train[test_index]
        yT, yt = self.y_train[train_index], self.y_train[test_index]
        rfe = SVM_RFE_DYNAMIC_STEP(n_features_to_select=stop, percentage=percentage)
        return self._svm_rfe(rfe, XT, yT, Xt, yt)

    def svm_rfe_dynamic_step_only(self, args):
        train_index, test_index, percentage, stop = args
        XT, Xt = self.X_train[train_index], self.X_train[test_index]
        yT, yt = self.y_train[train_index], self.y_train[test_index]
        rfe = SVM_RFE_DYNAMIC_STEP(n_features_to_select=stop, percentage=percentage)
        return self._svm_rfe_only(rfe, XT, yT, Xt, yt)

    def svm_rfe_sampling(self, args):
        stop = 1
        train_index, test_index, step, percentage = args
        XT, Xt = self.X_train[train_index], self.X_train[test_index]
        yT, yt = self.y_train[train_index], self.y_train[test_index]
        rfe = SVM_RFE_SAMPLING(n_features_to_select=1, step=step, percentage=percentage)
        print("HELLO", XT.shape)
        return self._svm_rfe(rfe, XT, yT, Xt, yt)