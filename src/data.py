from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    "transaction_data.csv": {"household_key", "basket_id", "day", "product_id", "quantity", "sales_value"},
    "product.csv": {"product_id"},
}


def _find_file(data_dir: Path, names: list[str]) -> Path | None:
    lowered = {p.name.lower(): p for p in data_dir.glob("*.csv")}
    return next((lowered[n.lower()] for n in names if n.lower() in lowered), None)


def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load Complete Journey CSVs, tolerating common filename variants."""
    root = Path(data_dir)
    aliases = {
        "transactions": ["transaction_data.csv", "transactions.csv"],
        "products": ["product.csv", "products.csv"],
        "demographics": ["hh_demographic.csv", "demographics.csv"],
        "campaigns": ["campaign_table.csv", "campaigns.csv"],
        "campaign_dates": ["campaign_desc.csv"],
    }
    found = {key: _find_file(root, names) for key, names in aliases.items()}
    if not found["transactions"] or not found["products"]:
        raise FileNotFoundError(
            f"Required transaction/product CSVs were not found in {root}. "
            "Run `python run_analysis.py --synthetic` or follow data/README.md."
        )
    frames = {key: pd.read_csv(path) for key, path in found.items() if path}
    for frame in frames.values():
        frame.columns = frame.columns.str.strip().str.lower()
        frame.rename(columns={"retail_disc": "retail_discount"}, inplace=True)
    raw_rows = len(frames["transactions"])
    tx = frames["transactions"]
    missing = REQUIRED["transaction_data.csv"] - set(tx.columns)
    if missing:
        raise ValueError(f"transaction_data.csv is missing columns: {sorted(missing)}")
    missing = REQUIRED["product.csv"] - set(frames["products"].columns)
    if missing:
        raise ValueError(f"product.csv is missing columns: {sorted(missing)}")
    frames = clean_data(frames)
    frames["transactions"].attrs["raw_rows"] = raw_rows
    return frames


def clean_data(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frames = {k: v.copy() for k, v in frames.items()}
    tx = frames["transactions"]
    numeric = ["household_key", "basket_id", "day", "product_id", "quantity", "sales_value"]
    for col in numeric:
        tx[col] = pd.to_numeric(tx[col], errors="coerce")
    tx = tx.dropna(subset=numeric).drop_duplicates()
    tx = tx[(tx.quantity > 0) & (tx.sales_value >= 0)]
    if "retail_discount" not in tx:
        tx["retail_discount"] = 0.0
    tx["retail_discount"] = pd.to_numeric(tx["retail_discount"], errors="coerce").fillna(0.0)
    tx["week_no"] = pd.to_numeric(tx.get("week_no", np.ceil(tx.day / 7)), errors="coerce")
    # 원본 DAY는 상대 일수이므로 임의의 달력 날짜를 부여하지 않습니다.
    if "date" in tx:
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    frames["transactions"] = tx
    products = frames["products"].drop_duplicates("product_id").copy()
    for col in ["department", "commodity_desc", "sub_commodity_desc", "brand"]:
        if col not in products:
            products[col] = "UNKNOWN"
        products[col] = products[col].fillna("UNKNOWN").astype(str)
    frames["products"] = products
    return frames


def make_synthetic_data(seed: int = 42, n_households: int = 320, n_products: int = 90) -> dict[str, pd.DataFrame]:
    """Deterministic retail fixture with learnable customer/category preferences."""
    rng = np.random.default_rng(seed)
    categories = np.array(["FROZEN", "SNACKS", "BEVERAGE", "DAIRY", "MEAT", "PANTRY"])
    brands = np.array(["BRAND_A", "NATIONAL", "VALUE", "PREMIUM"])
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "department": "GROCERY",
        "commodity_desc": np.resize(categories, n_products),
        "sub_commodity_desc": [f"SUB_{i % 18:02d}" for i in range(n_products)],
        "brand": np.resize(brands, n_products),
    })
    product_price = rng.uniform(2.0, 18.0, n_products)
    rows, basket_id = [], 1
    for hh in range(1, n_households + 1):
        fav = rng.integers(0, len(categories), size=2)
        discount_affinity = rng.beta(2, 3)
        n_baskets = int(rng.integers(12, 36))
        days = np.sort(rng.choice(np.arange(1, 366), size=n_baskets, replace=False))
        for day in days:
            size = int(rng.integers(2, 8))
            weights = np.ones(n_products)
            weights[np.isin(np.arange(n_products) % len(categories), fav)] *= 5
            picks = rng.choice(n_products, size=size, replace=False, p=weights / weights.sum())
            for idx in picks:
                disc = float(rng.random() < discount_affinity) * rng.uniform(0.05, 0.35) * product_price[idx]
                qty = int(rng.integers(1, 4))
                rows.append((hh, basket_id, day, idx + 1, qty, round((product_price[idx] - disc) * qty, 2), round(-disc * qty, 2), int(np.ceil(day / 7))))
            basket_id += 1
    transactions = pd.DataFrame(rows, columns=["household_key", "basket_id", "day", "product_id", "quantity", "sales_value", "retail_discount", "week_no"])
    transactions["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(transactions.day - 1, unit="D")
    demographics = pd.DataFrame({
        "household_key": np.arange(1, n_households + 1),
        "AGE_DESC": rng.choice(["25-34", "35-44", "45-54", "55-64", "65+"], n_households),
        "INCOME_DESC": rng.choice(["Under 25K", "25-49K", "50-74K", "75K+"], n_households),
        "HH_COMP_DESC": rng.choice(["Single", "Couple", "Family"], n_households),
    })
    campaigns = pd.DataFrame({
        "household_key": rng.choice(np.arange(1, n_households + 1), size=n_households // 2, replace=False),
        "CAMPAIGN": rng.integers(1, 6, n_households // 2),
    })
    return clean_data({"transactions": transactions, "products": products, "demographics": demographics, "campaigns": campaigns})
