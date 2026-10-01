from __future__ import annotations

import argparse, json
from pathlib import Path
import joblib
import pandas as pd

from src.data import load_data, make_synthetic_data
from src.features import build_customer_features, feature_dictionary
from src.prediction import fit_propensity_model, make_propensity_target
from src.recommender import ItemCFRecommender, PopularityRecommender, evaluate_model, temporal_holdout
from src.visualization import render_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--plots-only", action="store_true", help="저장된 CSV로 그래프만 다시 생성합니다")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        render_plots(pd.read_csv(out / "recommendation_metrics.csv"), pd.read_csv(out / "feature_importance.csv"), out)
        print("저장된 지표로 그래프 2개를 생성했습니다. 모델과 분석 지표는 변경하지 않았습니다.")
        return
    frames = make_synthetic_data(args.seed) if args.synthetic else load_data(args.data_dir)
    tx, products = frames["transactions"], frames["products"]
    campaigns = frames.get("campaigns")
    if campaigns is not None and "campaign_dates" in frames:
        campaigns = campaigns.merge(frames["campaign_dates"][["campaign", "start_day"]], on="campaign", how="left")

    train, test = temporal_holdout(tx)
    catalog = train.groupby("product_id").quantity.sum().nlargest(args.max_items).index
    catalog_train = train[train.product_id.isin(catalog)]
    model_rows = []
    for name, model in [("인기 상품 추천", PopularityRecommender()), ("구매 이력 기반 개인화 추천", ItemCFRecommender())]:
        model_rows.append({"model": name, **evaluate_model(model, catalog_train, test, k=10)})
    rec_metrics = pd.DataFrame(model_rows)
    rec_metrics.to_csv(out / "recommendation_metrics.csv", index=False)

    cutoff = int(tx.day.max() - 28)
    features = build_customer_features(tx, products, cutoff, campaigns)
    features.to_csv(out / "customer_features_sample.csv", index=False)
    active_dictionary = feature_dictionary()[feature_dictionary().feature.isin(features.columns)]
    active_dictionary.to_csv(out / "feature_dictionary.csv", index=False)
    X, y = make_propensity_target(features, tx, cutoff)
    propensity_model, prop_metrics, importance, scored = fit_propensity_model(X, y, args.seed)
    joblib.dump(propensity_model, out / "propensity_model.joblib")
    importance.to_csv(out / "feature_importance.csv", index=False)
    top_targets = pd.concat([features[["household_key"]], scored[["propensity_score"]]], axis=1).sort_values("propensity_score", ascending=False).head(50)
    top_targets.to_csv(out / "target_customers.csv", index=False)
    segments = features[["household_key", "preferred_category", "discount_sensitivity"]].merge(top_targets, on="household_key")
    segments["discount_segment"] = pd.cut(segments.discount_sensitivity, [-1, .1, .25, 1], labels=["low", "medium", "high"])
    (segments.groupby(["preferred_category", "discount_segment"], observed=True)
     .agg(customers=("household_key", "size"), mean_propensity=("propensity_score", "mean"))
     .reset_index().sort_values("mean_propensity", ascending=False).to_csv(out / "target_segments.csv", index=False))

    render_plots(rec_metrics, importance, out)

    summary = {
        "data_source": "synthetic_fixture" if args.synthetic else "dunnhumby_complete_journey",
        "rows": int(len(tx)), "households": int(tx.household_key.nunique()), "products": int(tx.product_id.nunique()),
        "raw_rows": int(tx.attrs.get("raw_rows", len(tx))),
        "product_table_rows": int(len(products)),
        "demographic_rows": int(len(frames.get("demographics", []))),
        "campaign_rows": int(len(frames.get("campaigns", []))),
        "day_min": int(tx.day.min()), "day_max": int(tx.day.max()), "cutoff_day": cutoff,
        "feature_count": int(features.shape[1] - 1),
        "recommendation_catalog": int(len(catalog)),
        "recommendation_users": int(test.household_key.nunique()),
        "recommendation_truth_coverage": float(test.product_id.isin(catalog).mean()),
        "recommendation": rec_metrics.round(4).to_dict("records"),
        "propensity": {k: round(v, 4) for k, v in prop_metrics.items()},
        "top_propensity_features": importance.head(5).round(4).to_dict("records"),
        "caveat": "고객을 학습용 75%와 평가용 25%로 나누고, 학습에 사용하지 않은 고객의 다음 28일 구매 여부로 성능을 평가했습니다.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
