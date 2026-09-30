from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def temporal_holdout(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """공통 기준일 이후 첫 구매일로 평가하여 고객 간 미래 정보 누수를 막습니다."""
    cutoff = int(transactions.day.max() - 28)
    train = transactions[transactions.day <= cutoff].copy()
    future = transactions[transactions.day > cutoff]
    first = future.groupby("household_key").day.transform("min")
    test = future[future.day.eq(first)].copy()
    eligible = train.household_key.unique()
    return train, test[test.household_key.isin(eligible)]


class PopularityRecommender:
    def fit(self, tx: pd.DataFrame):
        self.items_ = tx.groupby("product_id").quantity.sum().sort_values(ascending=False).index.to_list()
        self.seen_ = tx.groupby("household_key").product_id.apply(set).to_dict()
        return self

    def recommend(self, user: int, k: int = 10) -> list[int]:
        return [int(i) for i in self.items_][:k]


class ItemCFRecommender:
    """Implicit item-item collaborative filter using cosine similarity."""
    def fit(self, tx: pd.DataFrame):
        users = np.sort(tx.household_key.unique()); items = np.sort(tx.product_id.unique())
        self.user_to_idx = {v: i for i, v in enumerate(users)}
        self.item_to_idx = {v: i for i, v in enumerate(items)}
        self.idx_to_item = items
        rows = tx.household_key.map(self.user_to_idx).to_numpy()
        cols = tx.product_id.map(self.item_to_idx).to_numpy()
        vals = np.log1p(tx.quantity.to_numpy())
        self.matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
        self.similarity = cosine_similarity(self.matrix.T, dense_output=False)
        return self

    def recommend(self, user: int, k: int = 10) -> list[int]:
        if user not in self.user_to_idx:
            return []
        row = self.matrix.getrow(self.user_to_idx[user])
        scores = row @ self.similarity
        scores = np.asarray(scores.toarray()).ravel()
        # 식료품 추천에는 기존 구매 상품의 재구매도 포함합니다.
        top = np.argsort(-scores)[:k]
        return [int(self.idx_to_item[i]) for i in top if np.isfinite(scores[i])]


def ranking_metrics(actual: dict[int, set[int]], recommendations: dict[int, list[int]], k: int = 10) -> dict[str, float]:
    precision, recall, ndcg = [], [], []
    for user, truth in actual.items():
        pred = recommendations.get(user, [])[:k]
        hits = np.array([int(item in truth) for item in pred])
        precision.append(hits.sum() / k)
        recall.append(hits.sum() / len(truth) if truth else 0)
        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / np.log2(rank + 2) for rank in range(min(len(truth), k)))
        ndcg.append(dcg / ideal if ideal else 0)
    return {f"precision@{k}": float(np.mean(precision)), f"recall@{k}": float(np.mean(recall)), f"ndcg@{k}": float(np.mean(ndcg))}


def evaluate_model(model, train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> dict[str, float]:
    model.fit(train)
    actual = test.groupby("household_key").product_id.apply(set).to_dict()
    recs = {int(user): model.recommend(int(user), k) for user in actual}
    return ranking_metrics(actual, recs, k)
