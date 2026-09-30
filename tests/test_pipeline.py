import unittest

from src.data import make_synthetic_data, clean_data
from src.features import build_customer_features
from src.recommender import ItemCFRecommender, PopularityRecommender, evaluate_model, temporal_holdout


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = make_synthetic_data(seed=7, n_households=30, n_products=24)

    def test_features_are_point_in_time(self):
        tx = self.frames["transactions"]
        features = build_customer_features(tx, self.frames["products"], cutoff_day=180)
        expected = tx[tx.day <= 180].groupby("household_key").sales_value.sum()
        actual = features.set_index("household_key").monetary
        for household in actual.index:
            self.assertAlmostEqual(actual[household], expected[household])

    def test_recommenders_return_finite_metrics(self):
        train, test = temporal_holdout(self.frames["transactions"])
        for model in (PopularityRecommender(), ItemCFRecommender()):
            metrics = evaluate_model(model, train, test, k=5)
            self.assertTrue(all(0 <= value <= 1 for value in metrics.values()))


    def test_relative_day_does_not_invent_calendar(self):
        frames = {key: frame.copy() for key, frame in self.frames.items()}
        frames["transactions"] = frames["transactions"].drop(columns="date")
        cleaned = clean_data(frames)
        self.assertNotIn("date", cleaned["transactions"].columns)
        features = build_customer_features(cleaned["transactions"], cleaned["products"], cutoff_day=180)
        self.assertNotIn("weekend_ratio", features.columns)

    def test_holdout_uses_common_cutoff(self):
        tx = self.frames["transactions"]
        train, test = temporal_holdout(tx)
        cutoff = tx.day.max() - 28
        self.assertTrue((train.day <= cutoff).all())
        self.assertTrue((test.day > cutoff).all())


if __name__ == "__main__":
    unittest.main()
