# eda.py — clean & simple EDA for veori.csv
# Run: python eda.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DATA_PATH = "loan_dataset_malaysia1.csv"
DATA_PATH = "ve1.csv"

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- helpers ----------
def save_txt_info(df, path):
    with open(path, "w") as f:
        df.info(buf=f)

def sanitize(name):
    return str(name).replace("/", "_").replace("\\", "_")

def find_col(df, aliases):
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None

def iqr_mask(s):
    """Return boolean mask for IQR outliers (1.5×)."""
    s_clean = s.dropna()
    if s_clean.empty:
        return pd.Series(False, index=s.index)
    q1, q3 = s_clean.quantile(0.25), s_clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lower) | (s > upper)

def gaussian_kde_1d(x, grid, bw=None):
    """Simple Gaussian KDE using numpy only."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.zeros_like(grid)
    std = np.std(x)
    bw = bw or (std * n ** (-1/5))
    diff = (x[:, None] - grid[None, :]) / bw
    kern = np.exp(-0.5 * diff**2) / (np.sqrt(2*np.pi) * bw)
    return kern.mean(axis=0)

# ---------- load ----------
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise SystemExit(f"[ERROR] File not found: {DATA_PATH}")

print(f"Loaded {DATA_PATH} with shape {df.shape}")

# ---------- column resolution ----------
target_col = find_col(df, ["Loan_Approval_Status"])

cat_aliases = {
    "Gender": ["Gender"],
    "Marital_Status": ["Marital_Status", "Marital Status"],
    "Dependents": ["Dependents"],
    "Education": ["Education"],
    "Employment_Status": ["Employment_Status", "Employment Status"],
    "City/Town": ["City/Town", "City_Town", "City", "Town"],
    "Loan_History": ["Loan_History", "Loan History"],
}
num_aliases = {
    "Annual_Income": ["Annual_Income", "Annual Income"],
    "Loan_Amount_Requested": ["Loan_Amount_Requested", "Loan Amount Requested"],
    "Loan_Term": ["Loan_Term", "Loan Term"],
}

resolved_cats = {k: find_col(df, v) for k, v in cat_aliases.items() if find_col(df, v)}
resolved_nums = {k: find_col(df, v) for k, v in num_aliases.items() if find_col(df, v)}

# ---------- dataset info ----------
save_txt_info(df, os.path.join(OUTPUT_DIR, "dataset_info.txt"))

# ---------- summary stats & skewness ----------
selected_num_names = ["Annual_Income", "Loan_Amount_Requested", "Loan_Term"]
selected_cols = [resolved_nums[n] for n in selected_num_names if n in resolved_nums]
if selected_cols:
    df[selected_cols].describe().transpose().to_csv(
        os.path.join(OUTPUT_DIR, "summary_statistics_selected.csv"))
    df[selected_cols].skew().to_frame("skewness").to_csv(
        os.path.join(OUTPUT_DIR, "skewness_selected.csv"))

# ---------- class distribution (pie) ----------
if target_col:
    counts = df[target_col].value_counts(dropna=False)
    counts.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))
    plt.figure(figsize=(5, 5))
    labels = [f"{cls} ({cnt})" for cls, cnt in zip(counts.index, counts.values)]
    plt.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Loan Approval Status Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loan_approval_status_pie.png"))
    plt.close()

# ---------- categorical bar charts ----------
for nice_name, col in resolved_cats.items():
    vc = df[col].astype("string").fillna("<NA>").value_counts(dropna=False)
    plt.figure(figsize=(7, 4))
    bars = plt.bar(vc.index.astype(str), vc.values)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)
    plt.title(f"{nice_name} distribution")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"bar_{sanitize(nice_name)}.png"))
    plt.close()

# ---------- histograms (frequency + smooth line) ----------
for nice in ["Annual_Income", "Loan_Amount_Requested", "Loan_Term"]:
    col = resolved_nums.get(nice)
    if not col: continue
    s = df[col].dropna().values
    if len(s) == 0: continue
    plt.figure(figsize=(6.5,4.2))
    # frequency histogram
    counts, bins, _ = plt.hist(s, bins=30, color="skyblue", alpha=0.6)
    # smooth line (KDE)
    grid = np.linspace(np.min(s), np.max(s), 300)
    dens = gaussian_kde_1d(s, grid)
    # scale density to match frequency
    dens_scaled = dens * len(s) * (bins[1] - bins[0])
    plt.plot(grid, dens_scaled, color="navy", linewidth=2)
    plt.title(f"{nice} Distribution Histogram")
    plt.xlabel(nice)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{sanitize(nice)}.png"))
    plt.close()


# ---------- boxplots (with min/max/Q1/median/Q3 labels) ----------
for nice_name in ["Annual_Income", "Loan_Amount_Requested","Loan_Term"]:
    col = resolved_nums.get(nice_name)
    if not col:
        continue
    s = df[col].dropna().values
    if s.size == 0:
        continue

    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    vmin, vmax = np.min(s), np.max(s)

    plt.figure(figsize=(5.5, 6))
    plt.boxplot(s, vert=True, showfliers=True)
    plt.title(f"{nice_name} Box Plot")
    plt.ylabel(nice_name)

    # Add labels
    x = 1.1
    plt.text(x, vmin, f"Min = {vmin:.0f}", va="top")
    plt.text(x, q1, f"Q1 = {q1:.0f}", va="center")
    plt.text(x, q2, f"Median = {q2:.0f}", va="center")
    plt.text(x, q3, f"Q3 = {q3:.0f}", va="center")
    plt.text(x, vmax, f"Max = {vmax:.0f}", va="bottom")

    plt.xlim(0.8, 1.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"box_{sanitize(nice_name)}.png"))
    plt.close()

# ---------- IQR outliers (save separate files) ----------

loan_col = resolved_nums.get("Loan_Amount_Requested")
if loan_col:
    mask = iqr_mask(df[loan_col])
    if mask.any():
        df.loc[mask].to_csv(
            os.path.join(OUTPUT_DIR, "outlier_rows_loan_amount_requested.csv"), index=False)

loan_col = resolved_nums.get("Loan_Term")
if loan_col:
    mask = iqr_mask(df[loan_col])
    if mask.any():
        df.loc[mask].to_csv(
            os.path.join(OUTPUT_DIR, "outlier_rows_loan_term.csv"), index=False)

loan_col = resolved_nums.get("Annual_Income")
if loan_col:
    mask = iqr_mask(df[loan_col])
    if mask.any():
        df.loc[mask].to_csv(
            os.path.join(OUTPUT_DIR, "outlier_rows_annual_income.csv"), index=False)

print(f"EDA complete → outputs in '{OUTPUT_DIR}'")