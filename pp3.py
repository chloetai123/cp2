#!/usr/bin/env python3
# pp3.py — Preprocess into 4 variants in this order:
#   1) Handle outliers (for no_outliers: remove on FULL dataset; for with_outliers: skip)
#   2) Train/test split (stratified, random_state=42)
#   3) Handle class imbalance (SMOTE on TRAIN only or not)
# Outputs (scaled-only) + artifacts per variant for deployment inference.

from pathlib import Path
import json, random, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# ================== CONFIG (edit DATA_PATH to swap dataset) ==================
DATA_PATH    = Path("ve1.csv")       # <— change to your CSV, then rerun this script
OUT_ROOT     = Path("preprocess_outputs")
TARGET       = "Loan_Approval_Status"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Columns to use (Age intentionally excluded per your literature review)
CAT_COLS = ["Marital_Status","Dependents","Education","Employment_Status","City/Town"]
BIN_COLS = ["Gender","Loan_History"]
NUM_BASE = ["Annual_Income","Loan_Amount_Requested","Loan_Term"]  # + computed LTI
# ============================================================================

def std_text(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.replace(r"[-_/]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip().str.lower())

def map_binary(series: pd.Series, kind: str) -> pd.Series:
    if kind == "Gender":
        g = std_text(series)
        return pd.Series(np.where(g.isin(["male","m","1"]),1,
                           np.where(g.isin(["female","f","0"]),0,np.nan)),
                         index=series.index, dtype="float64")
    if kind == "Loan_History":
        t = pd.to_numeric(series, errors="coerce")
        if t.notna().mean() > 0.6: return t.astype("float64")
        h = std_text(series)
        return pd.Series(np.where(h.isin(["1","yes","y","true"]),1,
                           np.where(h.isin(["0","no","n","false"]),0,np.nan)),
                         index=series.index, dtype="float64")
    return series

def remove_outliers_iqr_full(df: pd.DataFrame, cols):
    """Remove outliers using IQR computed on the FULL dataset (pre-split)."""
    m = pd.Series(True, index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        m &= x.between(lo, hi) | x.isna()
    return df.loc[m].copy()

def build_design(train_df: pd.DataFrame, df_any: pd.DataFrame, ohe_categories: dict) -> pd.DataFrame:
    # one-hot aligned to TRAIN categories
    parts = []
    for c in CAT_COLS:
        cats = ohe_categories[c]
        D = pd.get_dummies(df_any[c], prefix=c, dtype=float)
        for v in cats:
            col = f"{c}_{v}"
            if col not in D.columns: D[col] = 0.0
        D = D[[f"{c}_{v}" for v in cats]]
        parts.append(D)
    Dcat = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_any.index)
    base = df_any[["Annual_Income","Loan_Amount_Requested","Loan_Term","LTI","Gender","Loan_History"]].copy()
    X = pd.concat([base, Dcat], axis=1)
    return X

def process_branch(df_branch: pd.DataFrame, outlier_key: str, summary: dict):
    """Split → OHE → median impute → scale → SMOTE/no_SMOTE → save CSVs + artifacts (incl. LTI clip)."""
    # 2) train/test split
    X = df_branch.drop(columns=[TARGET])
    y = df_branch[TARGET].astype(int)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # text normalize (safety)
    for c in CAT_COLS + BIN_COLS:
        if c in Xtr_raw: Xtr_raw[c] = std_text(Xtr_raw[c])
        if c in Xte_raw: Xte_raw[c] = std_text(Xte_raw[c])

    # OHE categories from TRAIN only
    ohe_categories = {c: sorted(Xtr_raw[c].dropna().unique().tolist()) for c in CAT_COLS}

    # design matrices aligned to TRAIN categories
    Xtr = build_design(Xtr_raw, Xtr_raw, ohe_categories)
    Xte = build_design(Xtr_raw, Xte_raw, ohe_categories)

    # TRAIN medians for imputation
    medians = {c: float(pd.to_numeric(Xtr[c], errors="coerce").median()) for c in Xtr.columns}
    for c, m in medians.items():
        Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(m)
        Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(m)

    # LTI clipping bounds from TRAIN (RAW LTI in Xtr_raw)
    q01 = float(pd.to_numeric(Xtr_raw["LTI"], errors="coerce").quantile(0.01))
    q99 = float(pd.to_numeric(Xtr_raw["LTI"], errors="coerce").quantile(0.99))
    lti_clip = {"q01": q01, "q99": q99}

    # scale (fit on TRAIN only)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    feats = list(Xtr.columns)

    # 3) handle class imbalance: no_smote + smote
    for sm_flag in ["no_smote", "smote"]:
        vdir = OUT_ROOT / outlier_key / sm_flag / "scaled"
        vdir.mkdir(parents=True, exist_ok=True)
        art_dir = vdir.parent / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)

        X_train_final = Xtr_s
        y_train_final = ytr
        if sm_flag == "smote":
            sm = SMOTE(random_state=RANDOM_STATE)
            X_train_final, y_train_final = sm.fit_resample(Xtr_s, ytr)

        # save CSVs
        pd.DataFrame(X_train_final, columns=feats).to_csv(vdir/"X_train.csv", index=False)
        pd.DataFrame(Xte_s,        columns=feats).to_csv(vdir/"X_test.csv",  index=False)
        pd.DataFrame({TARGET: y_train_final}).to_csv(vdir/"y_train.csv", index=False)
        pd.DataFrame({TARGET: yte}).to_csv(vdir/"y_test.csv", index=False)

        # artifacts (simple and complete)
        (art_dir/"ohe_categories.json").write_text(json.dumps(ohe_categories, indent=2))
        (art_dir/"feature_columns.json").write_text(json.dumps({"feature_columns": feats}, indent=2))
        (art_dir/"medians.json").write_text(json.dumps({"medians": medians}, indent=2))
        (art_dir/"lti_clip.json").write_text(json.dumps(lti_clip, indent=2))
        joblib.dump(scaler, art_dir/"scaler.pkl")

        summary["variants"][f"{outlier_key}/{sm_flag}"] = {
            "rows_total": int(len(df_branch)),
            "train_rows": int(len(y_train_final)),
            "test_rows": int(len(yte)),
            "n_features_after_encoding": int(len(feats)),
            "scaled_paths": {
                "X_train": str(vdir/"X_train.csv"),
                "X_test":  str(vdir/"X_test.csv"),
                "y_train": str(vdir/"y_train.csv"),
                "y_test":  str(vdir/"y_test.csv"),
            },
            "artifacts_dir": str(art_dir),
        }

def main():
    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

    df = pd.read_csv(DATA_PATH)

    # normalize City_Town naming
    if "City_Town" in df.columns and "City/Town" not in df.columns:
        df = df.rename(columns={"City_Town":"City/Town"})

    need = CAT_COLS + BIN_COLS + NUM_BASE + [TARGET]
    for c in need:
        if c not in df.columns: df[c] = np.nan

    # clean + map
    for c in CAT_COLS + BIN_COLS:
        df[c] = std_text(df[c])
    df["Gender"]       = map_binary(df["Gender"], "Gender")
    df["Loan_History"] = map_binary(df["Loan_History"], "Loan_History")
    for c in NUM_BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").round().clip(0,1).astype("Int64")

    # LTI
    df["LTI"] = df["Loan_Amount_Requested"] / (df["Annual_Income"] + 1e-6)

    summary = {"data_path": str(DATA_PATH), "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "variants": {}}

    # 1) outliers handled per branch
    df_with = df.copy()  # with_outliers: skip removal
    df_no   = remove_outliers_iqr_full(df.copy(), cols=NUM_BASE + ["LTI"])  # no_outliers: remove on FULL df

    process_branch(df_with, "with_outliers", summary)
    process_branch(df_no,   "no_outliers",  summary)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT/"summary.json").write_text(json.dumps(summary, indent=2))
    print("[DONE] Preprocessing complete (outliers → split → SMOTE).")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()