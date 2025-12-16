# thresholds.py
# Build eligibility thresholds from APPROVED loans and save to thresholds/thresholds.json

import json
from pathlib import Path
import pandas as pd
import numpy as np

# ========== USER SETTINGS ==========
data_path = "ve1.csv"                           # input dataset
output_path = "thresholds/thresholds.json"      # output JSON path

# Expected columns
CAT_COLS = ["Gender", "Marital_Status", "Education", "Employment_Status", "City/Town"]
NUM_COLS = ["Annual_Income", "Loan_Amount_Requested", "Loan_Term"]
DISCRETE_COL = "Dependents"
BINARY_COL = "Loan_History"          # 1=good, 0=bad (majority may differ)
TARGET_COL = "Loan_Approval_Status"  # 1=approved, 0=rejected
# ===================================


def _mode_safe(series: pd.Series):
    """Return the most frequent value safely."""
    vc = series.dropna().value_counts()
    if vc.empty:
        return None
    top_count = vc.iloc[0]
    top_vals = vc[vc == top_count].index.tolist()
    try:
        return sorted(top_vals)[0]
    except Exception:
        return str(top_vals[0])


def _ensure_cols(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_thresholds(df_approved: pd.DataFrame) -> dict:
    """Compute thresholds from approved cases only."""
    _ensure_cols(df_approved, CAT_COLS + NUM_COLS + [DISCRETE_COL, BINARY_COL])

    out = {
        "categorical": {},
        "discrete": {},
        "binary": {},
        "numerical": {}
    }

    # ---- Categorical: Top-1 (mode) among approved ----
    for c in CAT_COLS:
        out["categorical"][c] = _mode_safe(df_approved[c])

    # ---- Numerical: p25–p75 among approved ----
    for c in NUM_COLS:
        col = pd.to_numeric(df_approved[c], errors="coerce")
        p25 = float(col.quantile(0.25)) if col.notna().any() else None
        p75 = float(col.quantile(0.75)) if col.notna().any() else None
        out["numerical"][c] = {"p25": p25, "p75": p75}

    # ---- Discrete: majority ----
    dep = pd.to_numeric(df_approved[DISCRETE_COL], errors="coerce")
    dep_majority = _mode_safe(dep)
    out["discrete"][DISCRETE_COL] = {"majority": dep_majority}

    # ---- Binary: majority (0 or 1) ----
    bin_majority = _mode_safe(df_approved[BINARY_COL])
    out["binary"][BINARY_COL] = {"majority": bin_majority}

    return out


def save_thresholds(thr: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thr, f, indent=2, ensure_ascii=False)


def main():
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    _ensure_cols(df, [TARGET_COL])

    # Filter approved only
    approved = df[df[TARGET_COL].astype(float) == 1.0].copy()
    if approved.empty:
        raise ValueError("No approved rows found (Loan_Approval_Status == 1). Cannot build thresholds.")

    thr = build_thresholds(approved)
    save_thresholds(thr, Path(output_path))

    # Print concise summary
    print("[DONE] thresholds.json written →", output_path)
    print("\n[Summary] Categorical (Top-1 among approved):")
    for k, v in thr["categorical"].items():
        print(f"  - {k}: {v}")

    print("\n[Summary] Numerical bands (p25–p75 among approved):")
    for k, rng in thr["numerical"].items():
        print(f"  - {k}: p25={rng['p25']}, p75={rng['p75']}")

    print("\n[Summary] Discrete / Binary:")
    dep = thr["discrete"].get(DISCRETE_COL, {})
    print(f"  - {DISCRETE_COL}: majority={dep.get('majority')}")
    bh = thr["binary"][BINARY_COL]["majority"]
    print(f"  - {BINARY_COL}: majority={bh}")


if __name__ == "__main__":
    main()