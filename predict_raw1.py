#!/usr/bin/env python3
# predict_raw.py — original pipeline, now includes LTI in outputs

from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ---------------- CONFIG ----------------
MODELS_DIR = Path("models/xgb")
BEST_META  = MODELS_DIR / "best_variant.json"

MODEL_JSON = MODELS_DIR / "xgb_model.json"
MODEL_PKL  = MODELS_DIR / "xgb_model.pkl"

THRESHOLD = 0.50
INPUT_CSV  = Path("boundary_cases1.csv") #change this to boundary_cases.csv
OUTPUT_CSV = Path("preds.csv")

DEFAULT_ONE_ROW = {
    "Gender":"male",
    "Marital_Status":"single",
    "Dependents":"0",
    "Education":"graduate",
    "Employment_Status":"employed",
    "City/Town":"urban",
    "Annual_Income":78000,
    "Loan_History":1,
    "Loan_Amount_Requested":20000,
    "Loan_Term":36
}

# ---------------- HELPERS ----------------
def load_best_variant():
    if not BEST_META.exists():
        raise SystemExit("[FATAL] Missing models/xgb/best_variant.json")
    best = json.loads(BEST_META.read_text())
    return Path(best["path"])

def load_artifacts(vdir):
    arts = vdir.parent / "artifacts"
    feats = json.loads((arts / "feature_columns.json").read_text())["feature_columns"]
    cats  = json.loads((arts / "ohe_categories.json").read_text())
    scaler = joblib.load(arts / "scaler.pkl")
    lti_clip = None
    if (arts / "lti_clip.json").exists():
        j = json.loads((arts / "lti_clip.json").read_text())
        lti_clip = (float(j.get("lo", -1e9)), float(j.get("hi", 1e9)))
    return feats, cats, scaler, lti_clip

def load_model():
    if MODEL_JSON.exists() and xgb is not None:
        bst = xgb.Booster()
        bst.load_model(str(MODEL_JSON))
        return "booster", bst
    elif MODEL_PKL.exists():
        mdl = joblib.load(MODEL_PKL)
        return "sklearn", mdl
    else:
        raise SystemExit("[FATAL] No model found under models/xgb/")

def _norm_cat(s):
    return s.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

# ---------------- PREPROCESS ----------------
def preprocess(df_raw, feats, catsmap, scaler, lti_clip):
    df = df_raw.copy()

    # Normalize categoricals
    for c in ["Gender","Marital_Status","Dependents","Education","Employment_Status","City/Town"]:
        df[c] = _norm_cat(df[c])

    # Ensure numeric types
    for c in ["Annual_Income","Loan_History","Loan_Amount_Requested","Loan_Term"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute LTI (added)
    df["LTI"] = df["Loan_Amount_Requested"] / df["Annual_Income"]
    if lti_clip is not None:
        lo, hi = lti_clip
        df["LTI"] = df["LTI"].clip(lo, hi)

    df["Gender"] = (df["Gender"] == "male").astype(int)
    df["Loan_History"] = df["Loan_History"].fillna(0).astype(int).clip(0, 1)

    # Build feature matrix
    X_parts = []
    for num_col in ["Annual_Income","Loan_Amount_Requested","Loan_Term","LTI","Gender","Loan_History"]:
        if num_col in feats:
            X_parts.append(df[[num_col]])

    for base_col, values in catsmap.items():
        for v in values:
            cname = f"{base_col}_{v}"
            if cname in feats:
                X_parts.append((df[base_col] == v).astype(int).rename(cname))

    X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=df.index)
    X = X.reindex(columns=feats, fill_value=0)

    # use X, not X.values
    Xs_arr = scaler.transform(X)          # returns ndarray but uses the column names internally
    Xs = pd.DataFrame(Xs_arr, columns=feats, index=df.index)

    return Xs, df

# ---------------- PREDICT ----------------
def predict_df(df_raw, feats, cats, scaler, lti_clip, model_type, model, threshold=THRESHOLD):
    Xs, df_enriched = preprocess(df_raw, feats, cats, scaler, lti_clip)

    if model_type == "booster":
        dmat = xgb.DMatrix(Xs.values)
        pred_prob = model.predict(dmat, validate_features=False)
    else:
        proba = model.predict_proba(Xs.values)
        pred_prob = proba[:, 1]

    df_enriched["pred_prob"]  = pred_prob
    df_enriched["pred_label"] = (pred_prob >= threshold).astype(int)
    df_enriched["Gender"] = df_enriched["Gender"].map({1:"male",0:"female"})

    # Add LTI in output
    cols = [
        "Gender","Marital_Status","Dependents","Education","Employment_Status","City/Town",
        "Annual_Income","Loan_History","Loan_Amount_Requested","Loan_Term",
        "LTI","pred_prob","pred_label"
    ]
    return df_enriched[cols]

def predict_one(feats, cats, scaler, lti_clip, model_type, model, row, threshold=THRESHOLD):
    df = pd.DataFrame([row])
    out = predict_df(df, feats, cats, scaler, lti_clip, model_type, model, threshold).iloc[0]
    return {"pred_prob": float(out["pred_prob"]), "pred_label": int(out["pred_label"]), "LTI": float(out["LTI"])}

# ---------------- MAIN ----------------
def main():
    vdir = load_best_variant()
    feats, cats, scaler, lti_clip = load_artifacts(vdir)
    model_type, model = load_model()

    if INPUT_CSV.exists():
        df_raw = pd.read_csv(INPUT_CSV)
        expected = [
            "Gender","Marital_Status","Dependents","Education","Employment_Status","City/Town",
            "Annual_Income","Loan_History","Loan_Amount_Requested","Loan_Term"
        ]
        df_raw = df_raw[expected]
        out = predict_df(df_raw, feats, cats, scaler, lti_clip, model_type, model, THRESHOLD)
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"[OK] Read {len(df_raw)} rows → wrote {OUTPUT_CSV} (includes LTI, prob, label)")
    else:
        res = predict_one(feats, cats, scaler, lti_clip, model_type, model, DEFAULT_ONE_ROW, THRESHOLD)
        print(json.dumps(res, indent=2))
        print("[INFO] No input file found; ran single-row demo.")

if __name__ == "__main__":
    main()