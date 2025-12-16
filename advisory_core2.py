# advisory_core2.py - Generates financial advise, eligibility explanations, and post-application steps. 

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

# ============CONFIG ============

THRESHOLDS_PATH = Path("thresholds/thresholds.json")

# Feature groups
CAT_COLS = ["Gender", "Marital_Status", "Education", "Employment_Status", "City/Town"]
NUM_COLS = ["Annual_Income", "Loan_Amount_Requested", "Loan_Term"]
DISCRETE_DEP_COL = "Dependents"     
BINARY_LOAN_HISTORY_COL = "Loan_History"

# Document list 
DOCS_SALARIED = [
    "Application Form",
    "Copy of NRIC / Passport",
    "Latest 1-6 months salary slips",
    "Latest 6-12 months EPF statement",
    "Latest BE or EA form with official tax receipts"
]

DOCS_NON_SALARIED = [
    "Application Form",
    "Copy of NRIC / Passport",
    "Business registration certificate",
    "Latest B/BE form with official tax receipts",
    "Latest 6 months company/personal bank statement"
]


# ============THRESHOLDS LOADING ============

def load_thresholds(path: Path = THRESHOLDS_PATH) -> Dict[str, Any]:
    """
    Load thresholds.json produced by thresholds.py.
    If missing, return an empty dict so the rest of the flow still works.
    """
    if not path.exists():
        print(f"[WARN] thresholds file not found at {path}, using empty thresholds.")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] failed to load thresholds from {path}: {e}")
        return {}


def _safe_get(d: Dict[str, Any], key: str, default=None):
    """Small helper to avoid KeyError on nested dicts."""
    return d.get(key, default) if isinstance(d, dict) else default


# ============THRESHOLD COMPARISON ============
## Compare user input against thresholds 

def compare_to_thresholds(
    user_input: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[Dict[str, Any]]:

    flags: List[Dict[str, Any]] = []

    # 1) Categorical: flag if not top-1 (mode) 
    cat_thr = _safe_get(thresholds, "categorical", {})

    for col in CAT_COLS:
        val = user_input.get(col, None)
        entry = cat_thr.get(col, None)

        # entry may be a dict or a raw value
        if isinstance(entry, dict):
            thr_val = entry.get("top1") or entry.get("mode") or entry.get("majority") or entry.get("threshold")
        else:
            thr_val = entry

        flagged = False
        if thr_val is not None and val is not None:
            try:
                flagged = str(val).strip().lower() != str(thr_val).strip().lower()
            except Exception:
                flagged = False

        if flagged:
            reason = (
                f"Your category '{val}' is less common among approved applicants "
                f"compared to the typical category '{thr_val}', but approval is still "
                "possible depending on other financial indicators."
            )
        else:
            if thr_val is not None:
                reason = "Your category is in line with the most common pattern among approved applicants."
            else:
                reason = "No threshold available for this feature."

        flags.append({
            "feature": col,
            "input_value": val,
            "threshold_info": f"Top-1 category among approved: {thr_val}",
            "type": "categorical",
            "flagged": bool(flagged),
            "reason": reason,
        })

    # 2) Numerical: flag if falls outside IQR  
    num_thr = _safe_get(thresholds, "numerical", {})

    for col in NUM_COLS:
        val = user_input.get(col, None)
        entry = num_thr.get(col, None)

        lo = hi = None
        if isinstance(entry, dict):
            lo = entry.get("q1", None)
            hi = entry.get("q3", None)

        flagged = False
        if val is not None and lo is not None and hi is not None:
            try:
                v = float(val)
                lo_f = float(lo)
                hi_f = float(hi)
                flagged = (v < lo_f) or (v > hi_f)
            except ValueError:
                flagged = False

        if flagged:
            reason = (
                f"This value is outside the typical range [{lo}, {hi}] observed "
                "among approved applicants."
            )
        else:
            if lo is not None and hi is not None:
                reason = (
                    f"This value lies within the central range [{lo}, {hi}] "
                    "commonly observed among approved applicants."
                )
            else:
                reason = "No numerical threshold available for this feature."

        flags.append({
            "feature": col,
            "input_value": val,
            "threshold_info": f"Typical (IQR) range among approved: [{lo}, {hi}]",
            "type": "numerical",
            "flagged": bool(flagged),
            "reason": reason,
        })

    # 3) Discrete: flag if its not mode value
    dep_entry = _safe_get(thresholds, "discrete", {}).get(DISCRETE_DEP_COL, None)

    # dep_entry might be a dict like: { "type": "discrete", "method": "mode", "threshold": 3 }
    if isinstance(dep_entry, dict):
        dep_thr = dep_entry.get("threshold", None)
    else:
        dep_thr = dep_entry

    dep_val = user_input.get(DISCRETE_DEP_COL, None)

    flagged_dep = False
    if dep_thr is not None and dep_val is not None:
        try:
            flagged_dep = int(dep_val) != int(dep_thr)
        except (ValueError, TypeError):
            flagged_dep = False

    if flagged_dep:
        dep_reason = (
            f"Most approved applicants have {dep_thr} dependents, "
            f"but your input is {dep_val}."
        )
    else:
        if dep_thr is not None:
            dep_reason = (
                "Number of dependents is in line with the most common pattern among approved applicants."
            )
        else:
            dep_reason = "No threshold available for number of dependents."

    flags.append({
        "feature": DISCRETE_DEP_COL,
        "input_value": dep_val,
        "threshold_info": f"Mode among approved: {dep_thr}",
        "type": "discrete",
        "flagged": bool(flagged_dep),
        "reason": dep_reason,
    })

    # 4) Binary: flag if its not mode value
    bin_entry = _safe_get(thresholds, "binary", {}).get(BINARY_LOAN_HISTORY_COL, None)

    if isinstance(bin_entry, dict):
        bin_thr = bin_entry.get("threshold", None)
    else:
        bin_thr = bin_entry

    bin_val = user_input.get(BINARY_LOAN_HISTORY_COL, None)

    flagged_bin = False
    if bin_thr is not None and bin_val is not None:
        try:
            flagged_bin = int(bin_val) != int(bin_thr)
        except (ValueError, TypeError):
            flagged_bin = False

    if flagged_bin:
        if bin_thr == 1:
            reason = (
                f"Most approved applicants have a positive credit history (Loan_History={bin_thr}), "
                f"but your input is {bin_val}. Approval can still be possible if other factors "
                "are strong, but improving your credit record would help."
            )
        else:
            # typical is 0, user is 1 – still explain neutrally
            reason = (
                f"Most approved applicants have Loan_History={bin_thr}, "
                f"while your input is {bin_val}. This pattern is less common, "
                "but approval is still possible depending on other inputs."
            )
    else:
        if bin_thr is not None:
            reason = (
                f"Your credit history value matches the most common pattern "
                f"(Loan_History={bin_thr}) among approved applicants."
            )
        else:
            reason = "No threshold available for credit history."

    flags.append({
        "feature": BINARY_LOAN_HISTORY_COL,
        "input_value": bin_val,
        "threshold_info": f"Mode among approved: {bin_thr}",
        "type": "binary",
        "flagged": bool(flagged_bin),
        "reason": reason,
    })

    return flags


# ============ GPT PROMPT BUILDERS ============

def build_flag_blocks_for_gpt(flags: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for f in flags:
        if not f["flagged"]:
            continue  # only include flagged features

        feature = f["feature"]
        val = f["input_value"]
        thr = f["threshold_info"]
        reason = f["reason"]

        if f["type"] == "numerical":
            lines.append(
                f"Feature: {feature}\n"
                f"Input value: {val}\n"
                f"Required Threshold: {thr}\n"
                f"Advice: [One sentence suggesting how to improve]\n"
            )
        else:
            lines.append(
                f"Feature: {feature}\n"
                f"Input value: {val}\n"
                f"Required Threshold: {thr}\n"
                f"Note: {reason}\n"
            )

    return "\n".join(lines)


def build_rejected_prompt(user_input: Dict[str, Any], flags_block: str) -> str:

    base = (
        "You are a Malaysian financial advisor.\n"
        "A user's personal loan application was NOT approved.\n\n"
        "Below is the user's input profile:\n"
    )

    for k, v in user_input.items():
        base += f"- {k}: {v}\n"

    base += (
        "\nBased on comparison with typical approved applicants in Malaysia, "
        "the following features are below or outside the usual thresholds:\n\n"
    )
    base += flags_block
    base += (
        "\nFor each numerical feature, replace the placeholder Advice line with "
        "ONE clear, practical sentence on how the user can improve this aspect "
        "to increase future loan eligibility. Use Malaysian context where relevant "
        "(for example: RM amounts, local banking practices, realistic time frames), "
        "but do NOT mention any specific bank names or internal policies.\n"
        "For categorical, discrete, and binary features, keep the notes as they are "
        "and do not add extra advice unless it is simple, fair, and clearly applicable "
        "in Malaysia.\n"
    )

    return base


# ============DOCUMENT LOGIC (APPROVED) ============

def select_docs_for_user(user_input: Dict[str, Any]) -> List[str]:
    emp_raw = user_input.get("Employment_Status", "")
    emp = str(emp_raw).strip().lower()

    if emp in ["employed", "unemployed"]:
        return DOCS_SALARIED
    else:
        # self-employed, business owner, freelancer, etc.
        return DOCS_NON_SALARIED


def build_docs_checklist_text(doc_list: List[str]) -> str:
    lines = ["Checklist of Required Documents for applying personal loan in Malaysia:"]
    for i, d in enumerate(doc_list, start=1):
        lines.append(f"{i}. {d}")
    return "\n".join(lines)


# ============GPT-4 CALL WRAPPER ============

try:
    from openai import OpenAI  # type: ignore
    _openai_client = OpenAI()
except Exception:
    _openai_client = None


def call_gpt4(prompt: str) -> str:
    if _openai_client is None:
        return (
            "[GPT-4 not configured]\n"
            "This is the prompt that would be sent to the model:\n\n"
            + prompt
        )

    resp = _openai_client.chat.completions.create(
        model="gpt-4o-mini",  # you can switch to different model here
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, #model is accurate and flexible (not too strict and predictable)
    )
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return "[ERROR] GPT-4 response could not be parsed."


# ============MODEL LOADING (FROM predict_raw1.py) ============

_model_cache: Optional[
    Tuple[List[str], Dict[str, Any], Any, Optional[Tuple[float, float]], str, Any]
] = None


def _load_model_and_artifacts():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    try:
        from predict_raw1 import load_best_variant, load_artifacts, load_model
    except ImportError as e:
        raise ImportError(
            "advisory_core2.py: could not import required functions from predict_raw1.py. "
            "Make sure predict_raw1.py is in the same folder and defines "
            "load_best_variant, load_artifacts, and load_model."
        ) from e

    vdir = load_best_variant()
    feats, cats, scaler, lti_clip = load_artifacts(vdir)
    model_type, model = load_model()

    _model_cache = (feats, cats, scaler, lti_clip, model_type, model)
    return _model_cache


# ============ MAIN: RUN ADVISORY ============

def run_advisory(user_input: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Load model + artifacts
    feats, cats, scaler, lti_clip, model_type, model = _load_model_and_artifacts()

    # Predict
    try:
        from predict_raw1 import predict_one  # reuse existing helper
    except ImportError as e:
        raise ImportError(
            "advisory_core.py: could not import predict_one from predict_raw1.py. "
            "Ensure predict_raw1.py is in the same folder and defines "
            "predict_one(feats, cats, scaler, lti_clip, model_type, model, row, threshold=...)."
        ) from e

    pred = predict_one(feats, cats, scaler, lti_clip, model_type, model, user_input)
    pred_label = int(pred.get("pred_label", 0))
    pred_prob = float(pred.get("pred_prob", 0.0))
    lti_val = pred.get("LTI", None)

    # 2) Load thresholds
    thresholds = load_thresholds()

    docs: List[str] = []
    flags: List[Dict[str, Any]] = []
    gpt_prompt = ""
    advisory_text = ""

    if pred_label == 1:
        # ============ APPROVED: document checklist only (no GPT) ============
        docs = select_docs_for_user(user_input)
        advisory_text = build_docs_checklist_text(docs)
    else:
        # ============ NOT APPROVED: threshold comparison + GPT advisory ============
        flags = compare_to_thresholds(user_input, thresholds)
        flags_block = build_flag_blocks_for_gpt(flags)
        gpt_prompt = build_rejected_prompt(user_input, flags_block)
        advisory_text = call_gpt4(gpt_prompt)

    return {
        "pred_label": pred_label,
        "pred_prob": pred_prob,
        "LTI": lti_val,
        "documents": docs,          
        "flags": flags,             
        "gpt_prompt": gpt_prompt,   
        "advisory_text": advisory_text,  
    }

# ============ OPTIONAL LOCAL DEMO ============

if __name__ == "__main__":
    # Example rejected case
    example_user_rejected = {
        "Gender": "male",
        "Marital_Status": "single",
        "Dependents": 3,
        "Education": "high school",
        "Employment_Status": "employed",
        "City/Town": "urban",
        "Annual_Income": 40000,
        "Loan_History": 0,
        "Loan_Amount_Requested": 80000,
        "Loan_Term": 60,
    }

    thr = load_thresholds()
    flags_demo = compare_to_thresholds(example_user_rejected, thr)
    flags_block_demo = build_flag_blocks_for_gpt(flags_demo)
    prompt_demo = build_rejected_prompt(example_user_rejected, flags_block_demo)

    print("=== REJECTED CASE PROMPT PREVIEW ===")
    print(prompt_demo)
    print("\n")

    # Example approved case for documents
    example_user_approved = {
        "Gender": "female",
        "Marital_Status": "married",
        "Dependents": 1,
        "Education": "graduate",
        "Employment_Status": "employed",
        "City/Town": "urban",
        "Annual_Income": 90000,
        "Loan_History": 0,
        "Loan_Amount_Requested": 20000,
        "Loan_Term": 36,
    }
    docs_demo = select_docs_for_user(example_user_approved)
    print("=== APPROVED CASE DOCS CHECKLIST ===")
    print(build_docs_checklist_text(docs_demo))