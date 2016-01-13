import glob
import os
import pdb
from collections import OrderedDict
import pandas as pd
from numpy import zeros, ones


class UserFeatureTransformer(object):
    def __init__(self, data_path=None):
        """

        :param data_path: string, local data path to be used
        :return:
        """
        self.data = {}
        self.path = data_path
        self.transformed = False
        # defines a pipeline mapping columns (or stubs) to methods for transformation
        # feature_transfomers should be
        self._feature_transformers = [("creation_time", self._user_age),
                                      ("invited_by_user_id", self._bin_invited),
                                      ("cat", self._code_categorical),
                                      ("visits", self._visit_stats),
                                      ("name", self._drop_feature),
                                      ("email", self._drop_feature),
                                      ("org_id", self._drop_feature),
                                      ("last_session_creation_time", self._drop_feature),

                                      ("creation_time", self._drop_feature)]
        self._check_data_path()
        self._load_data()

    def handleError(function):
        """
        Decorator with basic exception handling

        :return: wrapped function
        """

        def handleExc(*args):
            try:
                return function(*args)
            except Exception:
                return False

        return handleExc

    def transform_features(self, key="users"):
        """

        Applies feature transformations specified in _feature_transformers

        :param key: the key specifying the subset of data to be transformed
        :return:
        """
        success = []
        print("transforming features...")
        success = [func(feature) for feature, func in self._feature_transformers]

        if all(success):
            print("all columns successfully transformed")
            self.transformed = True
        else:
            raise RuntimeError("not all features transformed")

    def _check_data_path(self):
        """
        Checks if a data path was provided otherwise defaults to data contained in package
        """
        if not self.path:
            dirname, fname = os.path.split(os.path.abspath(__file__))
            self.path = '/'.join(dirname.split('/')[:-1]) + "/data/"

        print("using data path: {0}".format(self.path))

    def _load_data(self):
        """
        Loads csv into DataFrames

        """
        try:
            # strip trailing / if exists
            path = self.path[:-1] if self.path[-1] == "/" else self.path
            data_files = glob.glob("{0}/*.csv".format(path))
            for csv in data_files:
                # with current files results in engagement and users
                data_key = csv.split("_")[-1].split(".csv")[0]
                self.data[data_key] = pd.DataFrame.from_csv(csv)
                print("loaded %s" % csv)

        except IOError as e:
            print("error occurred loading csv files: {0}".format(e.message))

    @handleError
    def _code_categorical(self, _, key="users", cat_columns=["creation_source"]):
        """

        :param key: string, dataframe to use
        :param cat_columns: list of column names to encode
        :return:

        """

        self.data[key] = pd.get_dummies(self.data[key], columns=cat_columns)
        return True

    @handleError
    def _user_age(self, *args):
        """
        Using last_session_creation_time and creation_time computes customer usage_age in days

        """
        users = self.data["users"]
        users["usage_tenure"] = zeros(len(users))
        for user_id, features in users.iterrows():
            features = features.fillna(0)
            user_creation = pd.Timestamp(features["creation_time"])
            if features["last_session_creation_time"]:
                last_sess = pd.Timestamp(pd.datetime.utcfromtimestamp(features["last_session_creation_time"]))
                usage_age = (last_sess - user_creation).days
            else:
                usage_age = 0

            users.loc[user_id, "usage_tenure"] = usage_age

        self.data["users"] = users
        return True

    @handleError
    def _visit_stats(self, *args):
        """
        Engineers visit related features including the avg. diff in hours between sessions and the total
        number of visits

        """
        engagement = self.data["engagement"]
        users = self.data["users"]
        engagement.index = engagement.index.map(lambda t: pd.Timestamp(t.strftime('%Y-%m-%d-%H')))
        gdp_users = engagement.groupby("user_id")
        print("computing intervals between user sessions...")
        diffed = gdp_users.apply(self._diff_user_visits, granularity=pd.Timedelta("1h"))
        inter_sess_avg = diffed.groupby("user_id").mean().drop("visited", axis=1).rename(
                columns={"delta": "intersession_mu"})
        tot_visits = diffed.groupby("user_id").sum().drop("delta", axis=1).rename(columns={"visited": "n_visits"})
        visit_stats = inter_sess_avg.join(tot_visits)
        users_stats = users.join(visit_stats, how="left").fillna(0)
        single_visits = users_stats["n_visits"] <= 1
        users_stats.loc[single_visits, "intersession_mu"] = 999.0 * ones(len(single_visits))
        self.data["users"] = users_stats
        return True

    @handleError
    def _drop_feature(self, feature, key="users"):
        """

        :param feature: string, which feature to drop
        :param key: string, which DF to drop from
        :return: boolean
        """
        self.data[key] = self.data[key].drop(feature, axis=1)
        return True

    @handleError
    def _bin_invited(self, feature, key="users"):
        """

        :param feature: what feature to binarize with NaN getting False and exists getting True
        :param key: which DF to apply against
        :return: boolean
        """
        data = self.data[key]
        data[feature] = data[feature].fillna(0)
        data.loc[data[feature] != 0, feature] = 1
        return True

    def _diff_user_visits(self, visits_by_user, granularity=pd.Timedelta("1d")):
        """

        :param visits_by_user: DF of visits for one user
        :param granularity : the time granularity to use when computing deltas between sessions
        :return: new DF with delta between visits/sessions
        """
        visits_by_user.loc[:, "tstamp"] = pd.Series(visits_by_user.index, index=visits_by_user.index)
        visits_by_user.loc[:, "delta"] = visits_by_user["tstamp"].diff().fillna(pd.Timedelta("0 days")) / granularity
        return visits_by_user.drop("tstamp", axis=1)
