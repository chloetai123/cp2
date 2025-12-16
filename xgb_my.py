# xgb_my.py — Malaysia synthetic model
# Pipeline: no_outliers → SMOTE → scaled

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# ================== CONFIG ==================

DATA_PATH    = Path("loan_dataset_malaysia1.csv")
OUT_DIR      = Path("models_my")
ART_DIR      = OUT_DIR / "artifacts_my"

TARGET       = "Loan_Approval_Status"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

CAT_COLS = ["Marital_Status", "Dependents", "Education", "Employment_Status", "City/Town"]
BIN_COLS = ["Gender", "Loan_History"]
NUM_BASE = ["Annual_Income", "Loan_Amount_Requested", "Loan_Term"]

PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    learning_rate=0.07,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    seed=RANDOM_STATE,
)
N_ROUNDS = 350
THRESH   = 0.50


# ================== HELPERS ==================

def std_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[-_/]", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
         .str.lower()
    )

def map_binary(series: pd.Series, kind: str) -> pd.Series:
    if kind == "Gender":
        g = std_text(series)
        return pd.Series(np.where(g.isin(["male","m","1"]),1,
                                  np.where(g.isin(["female","f","0"]),0,0)),
                         index=series.index, dtype="float64")

    if kind == "Loan_History":
        g = std_text(series)
        return pd.Series(np.where(g.isin(["1","yes","y","true"]),1,
                                  np.where(g.isin(["0","no","n","false"]),0,0)),
                         index=series.index, dtype="float64")

    return series

def remove_outliers_iqr_full(df: pd.DataFrame, cols):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= x.between(lo, hi)
    return df.loc[mask].copy()

def build_design(df_any: pd.DataFrame, ohe_categories: dict) -> pd.DataFrame:
    parts = []
    for c in CAT_COLS:
        cats = ohe_categories[c]
        D = pd.get_dummies(df_any[c], prefix=c, dtype=float)
        for v in cats:
            col = f"{c}_{v}"
            if col not in D.columns:
                D[col] = 0.0
        D = D[[f"{c}_{v}" for v in cats]]
        parts.append(D)

    Dcat = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_any.index)
    base = df_any[["Annual_Income","Loan_Amount_Requested","Loan_Term","LTI","Gender","Loan_History"]].copy()
    return pd.concat([base, Dcat], axis=1)


# ================== MAIN ==================

def main():

    if not DATA_PATH.exists():
        raise SystemExit(f"[ERROR] Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded Malaysia dataset: {len(df)} rows")

    # Standardize text
    for c in CAT_COLS + BIN_COLS:
        df[c] = std_text(df[c])

    # Map binaries
    df["Gender"]       = map_binary(df["Gender"], "Gender")
    df["Loan_History"] = map_binary(df["Loan_History"], "Loan_History")

    # Convert numerics
    for c in NUM_BASE:
        df[c] = pd.to_numeric(df[c])

    # Ensure target clean
    df[TARGET] = pd.to_numeric(df[TARGET]).clip(0,1).astype(int)

    # Compute LTI
    df["LTI"] = df["Loan_Amount_Requested"] / (df["Annual_Income"] + 1e-6)

    # OUTLIERS → REMOVE
    df_clean = remove_outliers_iqr_full(df, NUM_BASE + ["LTI"])
    print(f"[INFO] After outlier removal: {len(df_clean)} rows")

    # Train/test split
    X_raw = df_clean.drop(columns=[TARGET])
    y     = df_clean[TARGET]

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[INFO] Train={len(Xtr_raw)}, Test={len(Xte_raw)}")

    # OHE categories from TRAIN only
    ohe_categories = {c: sorted(Xtr_raw[c].unique().tolist()) for c in CAT_COLS}

    # Build design matrices
    Xtr = build_design(Xtr_raw, ohe_categories)
    Xte = build_design(Xte_raw, ohe_categories)

    # Compute LTI clipping bounds
    q01 = float(Xtr_raw["LTI"].quantile(0.01))
    q99 = float(Xtr_raw["LTI"].quantile(0.99))
    lti_clip = {"q01": q01, "q99": q99}

    Xtr["LTI"] = Xtr["LTI"].clip(q01, q99)
    Xte["LTI"] = Xte["LTI"].clip(q01, q99)

    # SMOTE
    print("[INFO] Applying SMOTE...")
    sm = SMOTE(random_state=RANDOM_STATE)
    Xtr_sm, ytr_sm = sm.fit_resample(Xtr, ytr)

    print("[INFO] Post-SMOTE class ratio:")
    print(pd.Series(ytr_sm).value_counts(normalize=True))

    # Scaling
    scaler = StandardScaler().fit(Xtr_sm)
    Xtr_s  = scaler.transform(Xtr_sm)
    Xte_s  = scaler.transform(Xte)

    feats = list(Xtr.columns)

    # XGBoost
    print("[INFO] Training XGBoost (no_outliers/smote/scaled)...")

    dtr = xgb.DMatrix(Xtr_s, label=ytr_sm, feature_names=feats)
    dte = xgb.DMatrix(Xte_s, label=yte,    feature_names=feats)

    bst = xgb.train(
        PARAMS,
        dtr,
        num_boost_round=N_ROUNDS,
        evals=[(dte, "valid")],
        verbose_eval=False,
    )

    # Predict
    proba = bst.predict(dte)
    pred  = (proba >= THRESH).astype(int)

    acc  = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec  = recall_score(yte, pred, zero_division=0)
    f1   = f1_score(yte, pred, zero_division=0)
    auc  = roc_auc_score(yte, proba)

    print("\n=== Malaysia Synthetic Model Performance ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    # Save artifacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    bst.save_model(str(OUT_DIR / "xgb_model_my.json"))

    (OUT_DIR / "features_my.json").write_text(
        json.dumps({"feature_columns": feats}, indent=2)
    )
    (ART_DIR / "ohe_categories_my.json").write_text(
        json.dumps(ohe_categories, indent=2)
    )
    (ART_DIR / "lti_clip_my.json").write_text(
        json.dumps(lti_clip, indent=2)
    )
    joblib.dump(scaler, ART_DIR / "scaler_my.pkl")

    print(f"\n[SAVED] Malaysia model → {OUT_DIR/'xgb_model_my.json'}")
    print(f"[SAVED] Artifacts → {ART_DIR}")


if __name__ == "__main__":
    main()