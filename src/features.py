from __future__ import annotations

import numpy as np
import pandas as pd


def build_customer_features(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    cutoff_day: int | None = None,
    campaigns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a point-in-time customer feature table without future leakage."""
    cutoff = int(cutoff_day or transactions.day.max())
    tx = transactions[transactions.day <= cutoff].merge(products[["product_id", "commodity_desc", "brand"]], on="product_id", how="left")
    basket = tx.groupby(["household_key", "basket_id"], as_index=False).agg(
        basket_value=("sales_value", "sum"), basket_day=("day", "max")
    )
    base = tx.groupby("household_key").agg(
        last_day=("day", "max"), frequency=("basket_id", "nunique"),
        monetary=("sales_value", "sum"), units=("quantity", "sum"),
        discount_amount=("retail_discount", lambda s: -s.clip(upper=0).sum()),
    )
    base["recency"] = cutoff - base.pop("last_day")
    base["avg_order_value"] = basket.groupby("household_key").basket_value.mean()
    base["discount_sensitivity"] = base.discount_amount / (base.monetary + base.discount_amount).replace(0, np.nan)
    intervals = basket.sort_values(["household_key", "basket_day"]).groupby("household_key").basket_day.apply(lambda s: s.diff().mean())
    base["purchase_interval"] = intervals
    if "date" in tx and tx.date.notna().any():
        base["weekend_ratio"] = tx.assign(weekend=tx.date.dt.dayofweek.ge(5)).groupby("household_key").weekend.mean()
    base["category_diversity"] = tx.groupby("household_key").commodity_desc.nunique()
    for field, out in [("commodity_desc", "preferred_category"), ("brand", "preferred_brand")]:
        pref = (tx.groupby(["household_key", field]).sales_value.sum().rename("value").reset_index()
                .sort_values(["household_key", "value"], ascending=[True, False]).drop_duplicates("household_key"))
        base[out] = pref.set_index("household_key")[field]
    if campaigns is not None and {"household_key", "start_day"}.issubset(campaigns.columns):
        exposed = campaigns[campaigns.start_day <= cutoff].groupby("household_key").size().rename("campaign_exposures")
        base = base.join(exposed)
    base["campaign_exposures"] = base.get("campaign_exposures", 0)
    numeric = base.select_dtypes(include="number").columns
    base[numeric] = base[numeric].replace([np.inf, -np.inf], np.nan).fillna(0)
    return base.reset_index()


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame([
        ("recency", "마지막 구매 이후 경과 일수", "매일"),
        ("frequency", "기준일까지의 고유 장바구니 수", "매일"),
        ("monetary", "누적 실구매 금액", "매일"),
        ("units", "누적 구매 수량", "매일"),
        ("discount_amount", "누적 할인 금액", "매일"),
        ("avg_order_value", "장바구니당 평균 구매 금액", "매일"),
        ("discount_sensitivity", "할인액을 할인 전 구매 금액으로 나눈 비율", "매일"),
        ("purchase_interval", "장바구니 간 평균 구매 간격(일)", "매주"),
        ("weekend_ratio", "주말에 구매한 상품 행의 비율", "매주"),
        ("category_diversity", "구매한 서로 다른 상품 카테고리 수", "매주"),
        ("preferred_category", "구매 금액이 가장 높은 상품 카테고리", "매주"),
        ("preferred_brand", "구매 금액이 가장 높은 브랜드 유형", "매주"),
        ("campaign_exposures", "캠페인 배정 횟수", "캠페인별"),
    ], columns=["feature", "definition", "refresh_cadence"])
