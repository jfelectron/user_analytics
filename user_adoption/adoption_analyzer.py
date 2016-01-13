import pdb

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from user_analytics.user_adoption.user_features import UserFeatureTransformer


class AdoptionAnalyzer(UserFeatureTransformer):
    """

    """

    def __init__(self, data_path=None, transform=False):
        """

        :param data_path: optional local data path to load data from
        :param transform: whether to transform data on init
        :return:
        """
        self.segmented = False
        super(AdoptionAnalyzer, self).__init__(data_path=data_path)
        if transform:
            self.transform_features()

    def score_adopted(self, min_visits=3, max_days=7, **kwargs):
        """
        :param min_visits: int, min visits to be considered adopted
        :param max_days: int, maximum days for min_visits
        :return:
        """
        if not self.transformed:
            print("transforming features before scoring...")
            self.transform_feaures()

        engagement = self.data["engagement"]
        # per definition of adopted, we only care about daily visits
        daily_engagement = self._daily_index(engagement)
        # group by unique users and sum visits
        print("grouping on user ids...")
        visit_counts = daily_engagement.groupby("user_id").sum()
        # find users that meet minimum visit criteria
        print("filtering users that qualify based on total visits...")
        qualifying_active = visit_counts[visit_counts["visited"] >= min_visits]
        print("joining data...")
        # use a left join on qualifying active, which masks non-qualifying with NaN for visits
        masked_users = daily_engagement.join(qualifying_active, on="user_id", how="left", lsuffix="_binary")
        # now we clean up joined df by dropping NaN and summed visited
        selected_visits = masked_users.dropna().drop(["visited"], axis=1)
        # qualify selected users by visits within the defined range of days
        print("scoring adoption based on visits per range of consecutive days...")
        scored_qualifying = self._collect_adopted(selected_rows=selected_visits, min_visits=min_visits,
                                                  max_days=max_days)
        print("segmenting users into adoption subsets...")
        return self._get_user_subsets(all_users=engagement, scored=scored_qualifying, **kwargs)

    def segment_engagement(self, adopted_params={"min_visits": 3, "max_days": 7},
                           cohorts=["adopted", "sporadic", "low"]):
        """

        :param adopted_params: dict, params that define an adopted user
        :param cohorts: the user segment labels to use
        :return: dict of DF keyd on cohorts
        """
        users_data = self.data["users"]
        users_data.index.name = "user_id"
        segmented_users = self.score_adopted(**adopted_params)
        user_segments = {k: users_data.join(segmented_users[k], how="inner", sort=True) for k in cohorts}
        self.segmented = True
        return user_segments

    def visualize_segments(self, user_segments):
        """
        Utility to visualize user segments for exploratory analysis
        :param user_segments: dict of user cohors
        """

        summary = []

        cont_summary = []

        non_cat = ["intersession_mu", "usage_tenure","n_visits","adopted"]

        for k, v in user_segments.iteritems():
            v["user_segment"] = np.repeat(k, len(v))
            percents = self._cat_percents(v.describe(), non_cat)
            percents["segment"] = k
            summary.append(percents)

        joined_users = pd.concat([v for v in user_segments.values()])

        summary_df = pd.DataFrame(summary)
        summary_long = pd.melt(summary_df, id_vars=["segment"],
                               value_vars=[feat for feat in summary_df.columns if feat != "segment"],
                               var_name="feature", value_name="percent")

        cat_feats = sns.barplot(x="feature", y="percent", hue="segment", data=summary_long)
        plt.setp(cat_feats.get_xticklabels(), rotation=90)
        cat_feats.set(xlabel="features", ylabel="percent of cohort")




    def _cat_percents(self, desc, drop_features):
        """
        :param desc: DF resulting from DataFrame.describe()
        :return: dict of percent representation for categorical features
        """

        percents = desc.loc["mean"].drop(drop_features) * 100
        return dict(percents)

    def _get_user_subsets(self, all_users, scored):
        """

        :param all_users: DF of all users
        :param scored: DF with adoption scoring
        :return:
        """
        # filter adopted users
        adopted_users = scored[scored.adopted == True]
        # filter qualifying on visits, but not strictly adopted
        qualifying_users = scored[scored.adopted == False]

        all_scored = self._merge_all(all_users, scored)

        # generate new DF for low actiivty users
        non_qualifying_users = pd.DataFrame.from_records(
                [{"user_id": x, "adopted": False} for x in set(all_scored.user_id) - set(scored.index)],
                index="user_id")
        # return users subsets
        return {"adopted": adopted_users, "sporadic": qualifying_users, "low": non_qualifying_users}

    def _merge_all(self, all_users, scored):
        """
        merges all_users with scored DF, reindexes post merge

        :param all_users: DF of all users
        :param scored: DF with adoption scoring
        :return:
        """
        scored["user_id"] = scored.index
        all_scored = pd.merge(all_users, scored, on="user_id", how="outer").fillna(False)
        all_scored.index = all_scored["index"]
        return all_scored.drop("index", axis=1)

    def _daily_index(self, df):
        """

        :param df: DF with timestamp index to be truncated to daily granularity
        :return:
        """
        df.index = df.index.map(lambda t: pd.Timestamp(t.strftime('%Y-%m-%d')))
        df.loc[:, "index"] = df.index
        df.drop_duplicates("index", keep="first")
        return df.drop("index", axis=1)

    def _collect_adopted(self, selected_rows, min_visits, max_days):
        """

        :param selected_rows: DF representing qualifying users
        :param min_visits: int, min visits to be considered adopted
        :param max_days: int, maximum days for min_visits
        :return:
        """
        # here we assume contiguous days rather than business days
        # XXX is primarily a enterprise SaaS platform an likely has highly periodic
        # usage aligning with the business week
        adopted = []
        gpd_users = selected_rows.groupby("user_id")
        return pd.DataFrame(
                {"adopted": gpd_users.apply(self._range_select_users, min_visits=min_visits, max_days=max_days)})

    def _range_select_users(self, visits, min_visits, max_days):
        """

        :param visits:
        :param min_visits:
        :param max_days:
        :return:
        """
        diffed_visits = self._diff_user_visits(visits)
        window_sums = pd.rolling_sum(diffed_visits["delta"], window=min_visits)
        return True if any(window_sums <= max_days) else False
