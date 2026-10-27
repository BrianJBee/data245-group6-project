"""Config-driven modelling helpers.

Pipeline and model-comparison loop adapted from Nelson, Fridhi & Wagner,
LMU CMSI 3300 HW4 (2022), reused with instructor approval. Changes: columns come
from the dataset config (no hard-coded names), class_weight='balanced' replaces
the backwards {'<=50K': 1.4}, and evaluation returns metrics instead of printing.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def split(df, cfg, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split; y is 1 for the positive label."""
    X = df[cfg["numeric_cols"] + cfg["categorical_cols"]]
    y = (df[cfg["target"]] == cfg["positive_label"]).astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


class S_Pred:
    """Preprocess (median-impute numerics, one-hot categoricals) + classifier."""

    def __init__(self, cfg, classifier=None, scale=False):
        num_steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:  # helps LogReg / MLP converge; trees ignore it
            num_steps.append(("scale", StandardScaler()))
        self.cfg = cfg
        self.preprocessor = ColumnTransformer([
            ("num", Pipeline(num_steps), cfg["numeric_cols"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["categorical_cols"]),
        ])
        if classifier is None:
            classifier = LogisticRegression(class_weight="balanced", max_iter=2000)
        self.pipeline = Pipeline([("preprocessor", self.preprocessor),
                                  ("classifier", classifier)])

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        return self

    def classify(self, X):
        return self.pipeline.predict(X)

    def score(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def test_model(self, X_test, y_test):
        names = [self.cfg["negative_label"], self.cfg["positive_label"]]
        return classification_report(y_test, self.classify(X_test),
                                      target_names=names, digits=3)

    def metrics(self, X_test, y_test):
        pred, proba = self.classify(X_test), self.score(X_test)
        return {"Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "ROC-AUC": roc_auc_score(y_test, proba)}

    def feature_names(self):
        return [n.split("__", 1)[1] for n in
                self.pipeline.named_steps["preprocessor"].get_feature_names_out()]


def default_models():
    """The HW4 comparison set, with a single, fast MLP config."""
    return {
        "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=2000), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=10, class_weight="balanced",
                                                 random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                                 class_weight="balanced", n_jobs=-1,
                                                 random_state=RANDOM_STATE), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=150,
                                                         random_state=RANDOM_STATE), False),
        "MLP (64-32)": (MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                      max_iter=200, random_state=RANDOM_STATE), True),
    }


def compare_models(cfg, X_train, y_train, X_test, y_test, models=None,
                   max_train_rows=None):
    """Fit each model, return (results DataFrame sorted by F1, fitted dict)."""
    models = models or default_models()
    if max_train_rows and len(X_train) > max_train_rows:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_train_rows, stratify=y_train,
            random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, (clf, scale) in models.items():
        t0 = time.time()
        m = S_Pred(cfg, clf, scale=scale).fit(X_train, y_train)
        rows.append({"Model": name, **m.metrics(X_test, y_test),
                     "Fit time (s)": time.time() - t0})
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return res, fitted


def group_rates(y_true, y_pred, groups, min_n=50):
    """Positive-prediction rate, base rate and TPR per group (fairness peek)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred),
                      "g": np.asarray(groups)})
    out = d.groupby("g").apply(lambda t: pd.Series({
        "n": len(t), "Base rate": t["y"].mean(),
        "Predicted positive rate": t["p"].mean(),
        "TPR": t.loc[t["y"] == 1, "p"].mean() if (t["y"] == 1).any() else np.nan,
    }), include_groups=False)
    out["n"] = out["n"].astype(int)
    return out[out["n"] >= min_n].sort_values("n", ascending=False)
