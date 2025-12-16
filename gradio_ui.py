#gradio_ui.py - Develop the user interface using Gradio to connect loan approval prediction system and financial advisory system. 

from typing import Dict, Any, Tuple

import gradio as gr
from advisory_core2 import run_advisory


# ---------------- HELPER: WRAP PIPELINE FOR GRADIO ---------------- #

def predict_and_advise(
    gender: str,
    marital_status: str,
    dependents: str,
    education: str,
    employment_status: str,
    city_town: str,
    annual_income: float,
    loan_history: str,
    loan_amount_requested: float,
    loan_term: int,
) -> Tuple[str, str, str, str]:

    # 1) Build user_input dict for the pipeline
    user_input: Dict[str, Any] = {
        "Gender": gender,
        "Marital_Status": marital_status,
        "Dependents": int(dependents),
        "Education": education,
        "Employment_Status": employment_status,
        "City/Town": city_town,  # match ve1.csv column name exactly
        "Annual_Income": float(annual_income),
        "Loan_History": int(loan_history),
        "Loan_Amount_Requested": float(loan_amount_requested),
        "Loan_Term": int(loan_term),
    }

    # 2) Run full advisory logic (prediction + thresholds + GPT-4 / fallback)
    result = run_advisory(user_input)

    pred_label = int(result.get("pred_label", 0))
    pred_prob = float(result.get("pred_prob", 0.0))
    docs = result.get("documents", []) or []
    advisory_text = result.get("advisory_text", "") or ""

    # 3) Decision text
    decision_text = "APPROVED" if pred_label == 1 else "NOT APPROVED"

    # 4) Probability text
    prob_text = f"{pred_prob:.4f}"

    # 5) Document checklist (only populated for approved cases)
    if docs:
        lines = ["Document checklist:"]
        for i, d in enumerate(docs, start=1):
            lines.append(f"{i}. {d}")
        docs_text = "\n".join(lines)
    else:
        docs_text = "No document checklist (loan not approved or not applicable)."

    # 6) Advisory behaviour:
    if pred_label == 1:
        advisory_out = ""   
    else:
        advisory_out = advisory_text if advisory_text else "No advisory text generated."

    return decision_text, prob_text, docs_text, advisory_out


# ---------------- BUILD GRADIO APP ---------------- #

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Personal Loan Approval & Advisory System") as demo:
        gr.Markdown(
            """
            # Personal Loan Approval & Advisory System

            Enter your details to:
            1. Predict your personal loan approval.
            2. See the predicted approval probability.
            3. If approved: A document checklist will be provided for your preparation.
            4. If not approved: An advisory and explanation will be provided..
            """
        )

        with gr.Row():
            # Left column: categorical features
            with gr.Column():
                gender = gr.Dropdown(
                    label="Gender",
                    choices=["male", "female"],
                    value=None,
                )
                marital_status = gr.Dropdown(
                    label="Marital Status",
                    choices=["single", "married", "divorced"],
                    value=None,
                )
                dependents = gr.Dropdown(
                    label="Number of Dependents",
                    choices=["0", "1", "2", "3"],
                    value="0",
                )
                education = gr.Dropdown(
                    label="Education",
                    choices=["high school", "graduate", "postgraduate"],
                    value=None,
                )
                employment_status = gr.Dropdown(
                    label="Employment Status",
                    choices=["employed", "self-employed", "unemployed"],
                    value=None,
                )
                city_town = gr.Dropdown(
                    label="City / Town",
                    choices=["urban", "suburban", "rural"],
                    value=None,
                )

            # Right column: numeric + binary features
            with gr.Column():
                annual_income = gr.Number(
                    label="Annual Income (RM)",
                    value=None,
                    precision=0,
                )
                loan_history = gr.Dropdown(
                    label="Loan History (0 = bad / no history, 1 = good)",
                    choices=["0", "1"],
                    value=None,
                )
                loan_amount_requested = gr.Number(
                    label="Loan Amount Requested (RM)",
                    value=None,
                    precision=0,
                )
                loan_term = gr.Number(
                    label="Loan Term (months)",
                    value=None,
                    precision=0,
                )

                predict_btn = gr.Button("Predict & Get Advisory", variant="primary")

        # Outputs row
        with gr.Row():
            with gr.Column():
                decision_out = gr.Textbox(
                    label="Loan Decision",
                    interactive=False,
                )
                prob_out = gr.Textbox(
                    label="Predicted Approval Probability",
                    interactive=False,
                )
                docs_out = gr.Textbox(
                    label="Document Checklist (if approved)",
                    lines=10,
                    interactive=False,
                )

            with gr.Column():
                advisory_out = gr.Textbox(
                    label="Advisory / Explanation",
                    lines=18,
                    interactive=False,
                )

        # Wire button to function
        predict_btn.click(
            fn=predict_and_advise,
            inputs=[
                gender,
                marital_status,
                dependents,
                education,
                employment_status,
                city_town,
                annual_income,
                loan_history,
                loan_amount_requested,
                loan_term,
            ],
            outputs=[
                decision_out,
                prob_out,
                docs_out,
                advisory_out,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()