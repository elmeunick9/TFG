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
    def __init__(self, step=4, percentage=[0.8, 0.2], C=0.1):
        self.step = step
        self.percentage = percentage
        self.C = C

    def fit(self, X0, y, test=()):
        self.scores_ = {}
        self.wscores_ = {}
        self.ascores_ = {}
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
            estimator = LinearSVC(C=self.C, max_iter=5000, dual=False)
            estimator.fit(X, y)

            # Get importance and rank them
            importances = estimator.coef_ ** 2
            ranks = np.argsort(importances)

            # Flatten ranks, required for Multi-Class Classification.
            ranks = np.ravel(ranks)

            # Record score
            cfs = np.sum(support_) - 1
            self.scores_[cfs]  = estimator.score(X,y)
            if test: self.test_scores_[cfs] = estimator.score(test[0][:, features], test[1])

            # Eliminate the worse feature
            threshold = min(self.step, np.sum(support_))
            for i in range(0, threshold):
                selected_feature = features[ranks[i]]
                support_[selected_feature] = False
                ranking_[np.logical_not(support_)] += 1

            self.wscores_[cfs] = np.ravel(np.sort(importances))

        # Set final attributes
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        self.ascores_ = self.cost(self.wscores_, self.scores_, self.percentage[0], self.percentage[1], n_features)

        return self

    def cost(self, wscores, scores, w1, w2, n_features):
        def extract(q, q_max):
            #print('MINMAX (Q)', np.min(q), np.max(q))
            rf = len(q)
            s = [q[0]]
            for i in range(1, len(q)):
                s += [s[i - 1] + q[i]]

            s_max = np.max(s)
            #print(s)
            c = []
            for e in s:
                acc = (1 - e / s_max) * (q_max - 0.5) + 0.5
                #c += [w1 * (1 - acc) + w2 * (rf - len(c)) / n_features]
                c += [acc]

            return {rf - i: e for i, e in enumerate(c)}

        C = {}
        for i in scores.keys():
            C[i] = extract(wscores[i], scores[i])

        keys = list(C.keys())
        c = {}
        for i in range(0, len(keys)):
            a = keys[i]
            b = keys[i + 1] if len(keys) < i + 1 else 0
            c.update({k: v for k, v in C[a].items() if a > k and k > b})

        return c

