#!/usr/bin/env python3
# lr1.py — Train Logistic Regression on all 4 variants (scaled), compare by AUC, save best.

from pathlib import Path
import json, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

PRE = Path("preprocess_outputs")
OUT = Path("models/lr")
TARGET = "Loan_Approval_Status"
SEED = 42

VARIANTS = [
    "with_outliers/no_smote/scaled",
    "with_outliers/smote/scaled",
    "no_outliers/no_smote/scaled",
    "no_outliers/smote/scaled",
]

def load_variant(vpath: Path):
    Xtr = pd.read_csv(vpath/"X_train.csv")
    ytr = pd.read_csv(vpath/"y_train.csv")[TARGET].astype(int)
    Xte = pd.read_csv(vpath/"X_test.csv")
    yte = pd.read_csv(vpath/"y_test.csv")[TARGET].astype(int)
    assert list(Xtr.columns)==list(Xte.columns)
    return Xtr,ytr,Xte,yte

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    best = {"auc":-1,"name":None,"path":None,"model":None,"feats":None,"metrics":None}

    for rel in VARIANTS:
        vdir = PRE/rel
        if not vdir.exists():
            print(f"[WARN] Missing {vdir}")
            continue
        print(f"[TRAIN] {rel}")
        Xtr,ytr,Xte,yte = load_variant(vdir)
        feats = list(Xtr.columns)

        lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
        lr.fit(Xtr, ytr)
        proba = lr.predict_proba(Xte)[:,1]
        pred  = (proba>=0.50).astype(int)
        m = {
            "auc": float(roc_auc_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0))
        }
        rows.append({"variant":rel,"path":str(vdir),**m})
        if m["auc"]>best["auc"]:
            best.update({"auc":m["auc"],"name":rel,"path":str(vdir),"model":lr,"feats":feats,"metrics":m})

    if not rows:
        raise SystemExit("[FATAL] No variants found. Run pp3.py first.")

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_csv(OUT/"variant_comparison.csv", index=False)
    print("\n[COMPARISON by AUC]\n", df.to_string(index=False))

    joblib.dump(best["model"], OUT/"lr_model.pkl")
    (OUT/"features.json").write_text(json.dumps({"feature_columns": best["feats"]}, indent=2))
    (OUT/"best_variant.json").write_text(json.dumps({"variant": best["name"], "path": best["path"], "metrics": best["metrics"]}, indent=2))
    print(f"\n[SAVED] {OUT/'lr_model.pkl'} + features.json + best_variant.json")

if __name__=="__main__":
    main()