from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_propensity_target(features: pd.DataFrame, transactions: pd.DataFrame, cutoff_day: int, horizon_days: int = 28):
    future = transactions[(transactions.day > cutoff_day) & (transactions.day <= cutoff_day + horizon_days)]
    buyers = set(future.household_key)
    y = features.household_key.isin(buyers).astype(int)
    return features.drop(columns="household_key"), y


def fit_propensity_model(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    categorical = X.select_dtypes(exclude="number").columns.to_list()
    numeric = X.select_dtypes(include="number").columns.to_list()
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    model = Pipeline([("preprocess", prep), ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=.06, max_leaf_nodes=15, random_state=seed))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=seed, stratify=y
    )
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = {"roc_auc_holdout": float(roc_auc_score(y_test, test_proba)), "average_precision_holdout": float(average_precision_score(y_test, test_proba)), "positive_rate": float(y.mean())}
    importance = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=seed)
    imp = pd.DataFrame({"feature": X.columns, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    scored = X.copy(); scored["propensity_score"] = model.predict_proba(X)[:, 1]; scored["actual_purchase"] = y.to_numpy()
    return model, metrics, imp, scored
