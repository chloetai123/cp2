# eval_my.py evaluates the existing XGBoost model (trained on ve1.csv) using the Malaysian synthetic dataset: loan_dataset_malaysia1.csv

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============ CONFIG ============ #

DATA_PATH = Path("loan_dataset_malaysia1.csv")

MODEL_JSON = Path("models/xgb/xgb_model.json")
FEATURES_JSON = Path("models/xgb/features.json")
BEST_VARIANT_JSON = Path("models/xgb/best_variant.json")

TARGET_COL = "Loan_Approval_Status"

RAW_COLS = [
    "Gender",
    "Marital_Status",
    "Dependents",
    "Education",
    "Employment_Status",
    "City/Town",
    "Annual_Income",
    "Loan_History",
    "Loan_Amount_Requested",
    "Loan_Term",
]

CAT_COLS = [
    "Marital_Status",
    "Dependents",
    "Education",
    "Employment_Status",
    "City/Town",
]

# Gender and Loan_History are treated as numeric (0/1) after mapping
NUM_BASE_COLS = [
    "Annual_Income",
    "Loan_Amount_Requested",
    "Loan_Term",
    "LTI",           # computed in this script
    "Gender",        # mapped to 0/1
    "Loan_History",  # mapped to 0/1 or numeric
]

# Helpers for preprocessing to match training
def find_artifacts_dir_from_best() -> Path:
    """Read best_variant.json and derive the artifacts directory."""
    if not BEST_VARIANT_JSON.exists():
        raise SystemExit(f"[ERROR] best_variant.json not found: {BEST_VARIANT_JSON}")

    with open(BEST_VARIANT_JSON, "r") as f:
        info = json.load(f)
    # info["path"] is something like: preprocess_outputs/no_outliers/smote/scaled
    vdir = Path(info["path"])
    # artifacts folder is sibling of 'scaled' (or sibling of the variant leaf)
    art_dir = vdir.parent / "artifacts"

    if not art_dir.exists():
        raise SystemExit(f"[ERROR] Artifacts directory not found: {art_dir}")

    return art_dir

#Loading preprocessed artefacts
def load_artifacts():
    # 1) artifacts dir (from best_variant.json written by xgb3.py)
    art_dir = find_artifacts_dir_from_best()

    # 2) One-hot categories (saved by pp3.py)
    ohe_cat_path = art_dir / "ohe_categories.json"
    if not ohe_cat_path.exists():
        raise SystemExit(
            f"[ERROR] Expected ohe_categories.json not found in artifacts dir: {ohe_cat_path}"
        )
    with open(ohe_cat_path, "r") as f:
        ohe_categories = json.load(f)

    # 3) StandardScaler
    scaler_path = art_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise SystemExit(
            f"[ERROR] Expected scaler.pkl not found in artifacts dir: {scaler_path}"
        )
    scaler = joblib.load(scaler_path)

    # 4) LTI clipping range
    lti_low, lti_high = -np.inf, np.inf
    lti_clip_path = art_dir / "lti_clip.json"
    if lti_clip_path.exists():
        with open(lti_clip_path, "r") as f:
            lti_clip = json.load(f)

        # Support both new (q01/q99) and older (low/high) keys
        if "q01" in lti_clip or "q99" in lti_clip:
            lti_low = float(lti_clip.get("q01", lti_low))
            lti_high = float(lti_clip.get("q99", lti_high))
        else:
            lti_low = float(lti_clip.get("low", lti_low))
            lti_high = float(lti_clip.get("high", lti_high))

    # 5) feature order (trained design matrix columns)
    if not FEATURES_JSON.exists():
        raise SystemExit(f"[ERROR] features.json not found: {FEATURES_JSON}")
    with open(FEATURES_JSON, "r") as f:
        feats_info = json.load(f)
    feat_cols = feats_info["feature_columns"]

    # 6) XGBoost model (Booster)
    if not MODEL_JSON.exists():
        raise SystemExit(f"[ERROR] xgb_model.json not found: {MODEL_JSON}")
    bst = xgb.Booster()
    bst.load_model(str(MODEL_JSON))

    return {
        "art_dir": art_dir,
        "ohe_categories": ohe_categories,
        "scaler": scaler,
        "lti_low": lti_low,
        "lti_high": lti_high,
        "features": feat_cols,
        "model": bst,
    }

#Standardizing text
def std_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[-_/]", " ", regex=True)
         .str.strip()
         .str.lower()
    )

#Mapping binary columns
def map_binary(series: pd.Series, kind: str) -> pd.Series:
    if kind == "Gender":
        g = std_text(series)
        return pd.Series(
            np.where(
                g.isin(["male", "m", "1"]),
                1,
                np.where(g.isin(["female", "f", "0"]), 0, np.nan),
            ),
            index=series.index,
            dtype="float64",
        )

    if kind == "Loan_History":
        t = pd.to_numeric(series, errors="coerce")
        # If majority are numeric, just use them
        if t.notna().mean() > 0.6:
            return t.astype("float64")
        h = std_text(series)
        return pd.Series(
            np.where(
                h.isin(["1", "yes", "y", "true"]),
                1,
                np.where(h.isin(["0", "no", "n", "false"]), 0, np.nan),
            ),
            index=series.index,
            dtype="float64",
        )

    return series

#Preprocess dataset
def preprocess_raw_df(df_raw: pd.DataFrame, A: dict) -> pd.DataFrame:
    df = df_raw.copy()

    # Cleaning
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in synthetic dataset: {missing}")

    # Binary mappings (same logic as training)
    df["Gender"] = map_binary(df["Gender"], "Gender")
    df["Loan_History"] = map_binary(df["Loan_History"], "Loan_History")

    # Categorical text normalisation
    for c in CAT_COLS:
        df[c] = std_text(df[c])

    # Numerics
    df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")
    df["Loan_Amount_Requested"] = pd.to_numeric(df["Loan_Amount_Requested"], errors="coerce")
    df["Loan_Term"] = pd.to_numeric(df["Loan_Term"], errors="coerce")

    # Compute LTI
    with np.errstate(divide="ignore", invalid="ignore"):
        lti = df["Loan_Amount_Requested"] / df["Annual_Income"]
    lti = lti.replace([np.inf, -np.inf], np.nan)
    lti = lti.fillna(0.0)
    lti = lti.clip(A["lti_low"], A["lti_high"])
    df["LTI"] = lti

    # OHE
    cats_map = A["ohe_categories"]
    parts = []
    for c in CAT_COLS:
        cats = cats_map.get(c, [])
        # dummies from current data
        D = pd.get_dummies(df[c], prefix=c, dtype=float)
        # ensure every training category column exists
        for v in cats:
            col = f"{c}_{v}"
            if col not in D.columns:
                D[col] = 0.0
        # keep columns in training order
        if cats:
            D = D[[f"{c}_{v}" for v in cats]]
        parts.append(D)

    if parts:
        X_cat_df = pd.concat(parts, axis=1)
    else:
        X_cat_df = pd.DataFrame(index=df.index)

    # Numeric base (including LTI and binary numerics) 
    X_num = df[NUM_BASE_COLS].copy()

    # combine
    X = pd.concat([X_num, X_cat_df], axis=1)

    # reorder to match training feature order, fill missing with 0
    feat_cols = A["features"]
    X = X.reindex(columns=feat_cols, fill_value=0.0)

    # Scale and return DataFrame 
    X_scaled_arr = A["scaler"].transform(X)  
    X_scaled = pd.DataFrame(X_scaled_arr, columns=feat_cols, index=X.index)

    return X_scaled

#Evaluate trained model using synthetic dataset (Malaysia-context)
def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"[ERROR] Synthetic dataset not found: {DATA_PATH}")

    print(f"[INFO] Loading synthetic dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise SystemExit(f"[ERROR] Target column '{TARGET_COL}' not found in {DATA_PATH}")

    # features and target from synthetic dataset
    X_raw = df[RAW_COLS].copy()
    y_true = df[TARGET_COL].astype(int).values

    # load artifacts + model
    A = load_artifacts()

    # preprocess synthetic data to model feature space
    print("[INFO] Preprocessing synthetic data to match training pipeline...")
    Xs = preprocess_raw_df(X_raw, A)   # DataFrame, columns == A["features"]

    # Build DMatrix with matching feature_names, but skip feature validation
    dmat = xgb.DMatrix(Xs, feature_names=A["features"])
    y_prob = A["model"].predict(dmat, validate_features=False)

    # threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation on Malaysian Synthetic Dataset ===")
    print(f"Samples: {len(df)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\nConfusion Matrix [ [TN FP] [FN TP] ] :")
    print(cm)

    # optional: save per-row predictions
    out_df = df.copy()
    out_df["pred_prob"] = y_prob
    out_df["pred_label"] = y_pred
    out_path = Path("malaysia_eval_preds.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Row-level predictions written to: {out_path}")


if __name__ == "__main__":
    main()