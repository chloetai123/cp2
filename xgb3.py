#!/usr/bin/env python3
# xgb3.py — Train XGBoost on all 4 preprocessed variants, compare by AUC, save best model.

from pathlib import Path
import json, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

PRE = Path("preprocess_outputs")
OUT = Path("models/xgb")
TARGET = "Loan_Approval_Status"
SEED = 42

VARIANTS = [
    "with_outliers/no_smote/scaled",
    "with_outliers/smote/scaled",
    "no_outliers/no_smote/scaled",
    "no_outliers/smote/scaled",
]

PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    learning_rate=0.07,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    seed=SEED,
)
N_ROUNDS = 350
THRESH   = 0.50

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
    best = {"auc":-1,"name":None,"path":None,"bst":None,"feats":None,"metrics":None}

    for rel in VARIANTS:
        vdir = PRE/rel
        if not vdir.exists():
            print(f"[WARN] Missing {vdir}")
            continue
        print(f"[TRAIN] {rel}")
        Xtr,ytr,Xte,yte = load_variant(vdir)
        feats = list(Xtr.columns)

        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
        dte = xgb.DMatrix(Xte, label=yte, feature_names=feats)
        bst = xgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS, evals=[(dte,"valid")], verbose_eval=False)

        proba = bst.predict(dte)
        pred  = (proba>=THRESH).astype(int)
        m = {
            "auc": float(roc_auc_score(yte, proba)),
            "accuracy": float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0))
        }
        rows.append({"variant":rel,"path":str(vdir),**m})
        if m["auc"]>best["auc"]:
            best.update({"auc":m["auc"],"name":rel,"path":str(vdir),"bst":bst,"feats":feats,"metrics":m})

    if not rows:
        raise SystemExit("[FATAL] No variants found. Run pp3.py first.")

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_csv(OUT/"variant_comparison.csv", index=False)
    print("\n[COMPARISON by AUC]\n", df.to_string(index=False))

    # Save best model + metadata for inference
    best["bst"].save_model(str(OUT/"xgb_model.json"))
    (OUT/"features.json").write_text(json.dumps({"feature_columns": best["feats"]}, indent=2))
    (OUT/"best_variant.json").write_text(json.dumps({"variant": best["name"], "path": best["path"], "metrics": best["metrics"]}, indent=2))
    print(f"\n[SAVED] {OUT/'xgb_model.json'} + features.json + best_variant.json")

if __name__=="__main__":
    main()