import pdb
import os
from operator import itemgetter
import time
from pprint import pformat

import pandas as pd
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import LabelKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

from user_analytics.user_adoption.adoption_analyzer import AdoptionAnalyzer

HOME_DIR = os.getenv("HOME")


class AdoptionPredictor(AdoptionAnalyzer):
    def __init__(self, training_iters=1000, cv_folds=5, scoring_method="f1", n_jobs=4):
        super(AdoptionPredictor, self).__init__()
        self.clf = RandomForestClassifier(class_weight="balanced_subsample")
        self.cv_params = {"folds": cv_folds, "scorer": scoring_method}
        self.n_iter = training_iters
        self.n_jobs = n_jobs
        # define param search space for RandomizedSearchCV
        self.param_grid = {"max_depth": [3, 5, 7, 9, None],
                           "n_estimators": sp_randint(10, 100),
                           "max_features": ["sqrt", "log2"],
                           "min_samples_split": sp_randint(3, 10),
                           "min_samples_leaf": sp_randint(1, 10),
                           "criterion": ["gini", "entropy"]}

    def fit(self, drop_features=[], segments=["adopted", "sporadic", "low"]):
        """

        :param drop_features: list, which features to drop before fit
        :param segments: which user segments to consider while fitting
        :return:
        """
        if not self.transformed:
            self.transform_features()
        user_id, class_labels, features = self._prep_for_fit(drop_features, segments=["adopted", "sporadic", "low"])
        # this is a user based model, therefore we want to avoid including same user in Train and Test
        cv_strat = LabelKFold(user_id, n_folds=self.cv_params["folds"])
        # RandomSearch is vastly faster than GridCV with tolerable loss of optimization
        # NB: RandomForest doesn't generally require heavy parm optimization, this is somewhat for posterity here
        random_search = RandomizedSearchCV(self.clf, param_distributions=self.param_grid,
                                           n_iter=self.n_iter, cv=cv_strat,
                                           scoring=self.cv_params["scorer"], n_jobs=self.n_jobs)

        print("running random param search on {} ".format(self.clf.__class__.__name__))
        random_search.fit(features, class_labels)
        self._handle_result(random_search, list(features.columns))

    def _handle_result(self, random_search, feature_names):
        """
        Generating training and model validation report
        Saves best model and associated meta-data

        :param random_search: the object after a finished fit of RSCV
        :param feature_names: list, feature names used during fit
        :return:
        """
        feature_importances = random_search.best_estimator_.feature_importances_
        importance_tuples = sorted(zip(feature_names, feature_importances), key=itemgetter(1), reverse=True)
        # generate report
        top_scores = sorted(random_search.grid_scores_, key=itemgetter(1), reverse=True)[:5]
        self.best_models(top_scores, importance_tuples)
        best_score = top_scores[0]
        # store best params and feature importance
        self.best_model = {"clf": random_search.best_estimator_, "params": best_score.parameters,
                           "mean_score": best_score.mean_validation_score,
                           "feature_importance": importance_tuples}

    def best_models(self, top_scores, importances):
        """
        Writes training report to timestamped file

        :param top_scores: ordered grid_scores from random search
        :param importances: ordered list of (feature,importance)

        """
        with open("{}/adoption_random_search_{}.txt".format(HOME_DIR, int(time.time())), "w") as f:
            for i, score in enumerate(top_scores):
                f.write("Model with rank: {0} \n".format(i + 1))
                f.write("Mean {0} score: {1:.3f} (std: {2:.3f}) \n".format(
                        self.cv_params["scorer"],
                        score.mean_validation_score,
                        np.std(score.cv_validation_scores)))
                f.write("Parameters: {0} \n\n".format(score.parameters))

            f.write("Feature Importances: \n\n")
            f.write(pformat(importances))

    def _prep_for_fit(self, drop_features=[], segments=["adopted", "sporadic", "low"]):
        """
        Performs final munging before fitting

        :param drop_features: list, which features to drop before fit
        :param segments: which user segments to consider while fitting
        :return: array of user_id, array of class labels, data frame of features
        """
        if not self.segmented:
            self.user_segments = self.segment_engagement()
        all_users = self.user_segments
        training_users = pd.concat([all_users[segment] for segment in segments]).sort_index()
        user_id = training_users.index
        # cast boolean to [0,1]
        class_labels = training_users["adopted"] * 1
        features = training_users.drop("adopted", axis=1)
        if drop_features:
            features = features.drop(drop_features, axis=1)

        return user_id, class_labels, features
