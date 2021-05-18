import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn import preprocessing

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

class SVM_RFE_STOPCOND():
    def __init__(self, step=4, percentage=[0.8, 0.2]):
        self.step = step
        self.percentage = percentage

    def fit(self, X0, y, test=()):
        self.scores_ = {}
        self.wscores_ = {}
        self.test_scores_ = {}
        
        n_features = X0.shape[1]
            
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)
    
        q_max = None
        prev_scal_q = 1

        # np.sum(support_) is the number of selected features.
        # It starts at n_features and decreases every iteration.
        while np.sum(support_) > 0:
            
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

            # Record score
            self.scores_[np.sum(support_) - 1]  = estimator.score(X,y)
            if test: self.test_scores_[np.sum(support_) - 1] = estimator.score(test[0][:, features], test[1])

            # Eliminate the worse feature
            threshold = min(self.step, np.sum(support_))
            for i in range(0, threshold):
                selected_feature = features[ranks[i]]
                support_[selected_feature] = False
                ranking_[np.logical_not(support_)] += 1
                if not q_max: q_max = np.max(np.ravel(importances))
                q = np.ravel(importances)[ranks[i]] / q_max
                self.wscores_[np.sum(support_)] = 1 - q
                
            # Scalarization and exit condition
            i = np.sum(support_)
            scal_q = self.percentage[0] * q + self.percentage[1] * (i / n_features)
            if not prev_scal_q < scal_q - 0.001:
                prev_scal_q = scal_q
                self.selected_i = i
                # break

        # Set final attributes
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self
