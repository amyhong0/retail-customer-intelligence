import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import matplotlib.pyplot as plt

from src.data import make_synthetic_data, clean_data
from src.features import build_customer_features
from src.recommender import ItemCFRecommender, PopularityRecommender, evaluate_model, temporal_holdout
from src.visualization import render_plots


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

    def test_plot_labels_match_portfolio(self):
        metrics = pd.read_csv("outputs/recommendation_metrics.csv")
        importance = pd.read_csv("outputs/feature_importance.csv")
        self.assertEqual(metrics.model.tolist(), ["인기 상품 추천", "구매 이력 기반 개인화 추천"])
        before = metrics.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.visualization.plt.close"):
                render_plots(metrics, importance, Path(directory))
                axes = [plt.figure(number).axes[0] for number in plt.get_fignums()]
                comparison = next(ax for ax in axes if ax.get_title() == "추천 모델 성능 비교")
                self.assertEqual([tick.get_text() for tick in comparison.get_xticklabels()], metrics.model.tolist())
                self.assertIn("구매 예측에 중요했던 고객 특성", [ax.get_title() for ax in axes])
            plt.close("all")
            self.assertTrue((Path(directory) / "recommendation_comparison.png").is_file())
            self.assertTrue((Path(directory) / "feature_importance.png").is_file())
        pd.testing.assert_frame_equal(metrics, before)


if __name__ == "__main__":
    unittest.main()
