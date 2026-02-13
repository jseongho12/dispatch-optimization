# train_models.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


FEATURE_COLS = [
    "PU_HUB", "DO_HUB",
    "cap", "cap_log1p",
    "PU_TIME_IDX", "DO_TIME_IDX",
    "tw_slack_min",
    "pu_hour", "do_hour",
    "direct_time", "direct_dist",
    "is_frozen_order",
]

CAT_COLS = ["PU_HUB", "DO_HUB", "is_frozen_order"]  # 허브는 범주로 처리
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def build_preprocess():
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, NUM_COLS),
            ("cat", cat_tf, CAT_COLS),
        ],
        remainder="drop",
    )
    return pre


def train_classifier(df: pd.DataFrame):
    pre = build_preprocess()
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )
    model = Pipeline([("pre", pre), ("clf", clf)])
    return model


def train_regressor(df: pd.DataFrame):
    pre = build_preprocess()
    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        random_state=42,
    )
    model = Pipeline([("pre", pre), ("reg", reg)])
    return model


def main(data_path="ml_data/dataset.csv", out_dir="ml_models"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if "scenario_id" not in df.columns:
        raise ValueError("dataset must include scenario_id")

    # 그룹 분할: 같은 시나리오는 train/test에 섞이지 않게
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["scenario_id"]))

    tr = df.iloc[train_idx].reset_index(drop=True)
    te = df.iloc[test_idx].reset_index(drop=True)

    X_tr = tr[FEATURE_COLS]
    X_te = te[FEATURE_COLS]

    # ---- classifier (assigned / unassigned)
    y_tr = tr["y_assigned"].astype(int)
    y_te = te["y_assigned"].astype(int)

    clf = train_classifier(tr)
    clf.fit(X_tr, y_tr)

    p = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap = average_precision_score(y_te, p)

    joblib.dump(clf, os.path.join(out_dir, "clf_assigned.joblib"))

    print(f"[Classifier] ROC-AUC={auc:.4f}  AP={ap:.4f}")

    # ---- regressor (optional): y_delta_obj
    if "y_delta_obj" in df.columns and df["y_delta_obj"].notna().any():
        # 회귀는 assigned==1인 것만 학습하는게 보통 더 깔끔함
        tr_r = tr[tr["y_assigned"] == 1].copy()
        te_r = te[te["y_assigned"] == 1].copy()

        tr_r = tr_r[tr_r["y_delta_obj"].notna()]
        te_r = te_r[te_r["y_delta_obj"].notna()]

        if len(tr_r) >= 200 and len(te_r) >= 50:
            reg = train_regressor(tr_r)
            reg.fit(tr_r[FEATURE_COLS], tr_r["y_delta_obj"].astype(float))

            pred = reg.predict(te_r[FEATURE_COLS])
            mae = mean_absolute_error(te_r["y_delta_obj"].astype(float), pred)
            r2 = r2_score(te_r["y_delta_obj"].astype(float), pred)

            joblib.dump(reg, os.path.join(out_dir, "reg_delta_obj.joblib"))
            print(f"[Regressor] MAE={mae:.3f}  R2={r2:.4f}")
        else:
            print("[Regressor] Not enough regression labels to train robustly.")
    else:
        print("[Regressor] No y_delta_obj column (or all NaN). Skipping.")

    meta = {
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "data_path": data_path,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved models to: {out_dir}")


if __name__ == "__main__":
    main()
