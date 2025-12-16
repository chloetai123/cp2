# boundary.py — Compute the boundary cases 

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter

# ============ Paths ============
MODELS_DIR = Path("models/xgb")
BEST_META  = MODELS_DIR / "best_variant.json"
OUT_CSV    = Path("boundary_cases.csv")

# RAW schema expected by predict_raw.py (order matters)
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

# how many top values to keep for multi-class categoricals
TOP_K_MULTI = 2

def main():
    # 1) Locate best variant + artifacts
    if not BEST_META.exists():
        raise SystemExit("[FATAL] models/xgb/best_variant.json not found. Run xgb3.py first.")
    best = json.loads(BEST_META.read_text())
    vdir = Path(best["path"])              # .../scaled
    arts = vdir.parent / "artifacts"       # .../artifacts

    feat_path = arts / "feature_columns.json"
    cats_path = arts / "ohe_categories.json"
    scaler_p  = arts / "scaler.pkl"
    xtrain_p  = vdir / "X_train.csv"

    for p in (feat_path, cats_path, scaler_p, xtrain_p):
        if not p.exists():
            raise SystemExit(f"[FATAL] Missing artifact: {p}")

    feats   = json.loads(feat_path.read_text())["feature_columns"]
    catsmap = json.loads(cats_path.read_text())  # { "Marital_Status": [...], ... }
    scaler  = joblib.load(scaler_p)
    Xs_tr   = pd.read_csv(xtrain_p)

    if list(Xs_tr.columns) != feats:
        raise SystemExit("[FATAL] X_train.csv columns do not match feature_columns order.")

    # 2) Inverse-scale train to raw feature space (numerics back to units; dummies ~0/1)
    X_tr = pd.DataFrame(scaler.inverse_transform(Xs_tr.values), columns=feats)

    # 3) Numeric levels = min, median, max from TRAIN (raw space)
    NUM_COLS = ["Annual_Income", "Loan_Amount_Requested", "Loan_Term"]
    num_levels = {}
    for c in NUM_COLS:
        s = pd.to_numeric(X_tr[c], errors="coerce").dropna()
        levels = [float(s.min()), float(s.median()), float(s.max())]
        levels = [int(round(v)) for v in levels]  # keep them tidy
        num_levels[c] = levels

    # 4) Category selection
    def top_k_for_col(base_col: str, k: int = TOP_K_MULTI):
        vals = catsmap.get(base_col, [])
        if not vals:
            return []
        counts = []
        for v in vals:
            dcol = f"{base_col}_{v}"
            cnt = X_tr[dcol].sum() if dcol in X_tr.columns else 0.0
            counts.append((v, cnt))
        counts.sort(key=lambda t: (-t[1], str(t[0])))
        picked = [v for v, cnt in counts[:k] if cnt > 0]
        # fallback if all zero (rare)
        if not picked:
            picked = vals[:k]
        return picked

    top_marital = top_k_for_col("Marital_Status")
    top_dep     = top_k_for_col("Dependents")
    top_edu     = top_k_for_col("Education")
    top_emp     = top_k_for_col("Employment_Status")
    top_city    = top_k_for_col("City/Town")

    # Binaries: mode from TRAIN raw-space columns 'Gender' and 'Loan_History'
    def mode01(col: str) -> int:
        if col not in X_tr.columns:
            return 1
        s = (X_tr[col].round().clip(0, 1)).astype(int)
        cnt = Counter(s)
        return 1 if cnt[1] >= cnt[0] else 0

    gender_mode = mode01("Gender")              
    loanhist_mode = mode01("Loan_History")      
    gender_val = "male" if gender_mode == 1 else "female"

    # 5) Cross-product: categories (with gender & loan history fixed to mode) × numeric levels
    AI_levels  = num_levels["Annual_Income"]
    LAR_levels = num_levels["Loan_Amount_Requested"]
    LT_levels  = num_levels["Loan_Term"]

    rows = []
    for ms in top_marital:
        for dp in top_dep:
            for edu in top_edu:
                for emp in top_emp:
                    for city in top_city:
                        for ai in AI_levels:
                            for lar in LAR_levels:
                                for lt in LT_levels:
                                    rows.append({
                                        "Gender": gender_val,
                                        "Marital_Status": ms,
                                        "Dependents": str(dp),  
                                        "Education": edu,
                                        "Employment_Status": emp,
                                        "City/Town": city,
                                        "Annual_Income": int(ai),
                                        "Loan_History": int(loanhist_mode),
                                        "Loan_Amount_Requested": int(lar),
                                        "Loan_Term": int(lt),
                                    })

    df = pd.DataFrame(rows)[RAW_COLS]
    df.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote boundary dataset with {len(df)} rows → {OUT_CSV}")
    print("[NUMERIC LEVELS]")
    for k, v in num_levels.items():
        print(f"  {k}: {v}  (min, median, max)")
    print("[CATEGORIES USED]")
    print(f"  Gender(mode)={gender_val}, Loan_History(mode)={loanhist_mode}")
    print(f"  Marital_Status={top_marital}")
    print(f"  Dependents={top_dep}")
    print(f"  Education={top_edu}")
    print(f"  Employment_Status={top_emp}")
    print(f"  City/Town={top_city}")

if __name__ == "__main__":
    main()