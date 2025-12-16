**Loan Approval Prediction and Financial Advising Using AI**

This repository contains the source code for **Capstone Project 2**, which focuses on building an AI-based system for personal loan approval prediction and financial advisory system.

Project structure: 
1. Loan approval prediction system
   - EDA.py - Exploratory data analysis
   - pp3.py - Data preprocessing and feature preparation
   - LR1.py - Logistic Regression model
   - xgb3.py - XGBoost model
   - xgb_my.py - Malaysia-specific XGBoost model
   - eval_my.py - Cross-domain testing
   - b1.py - Boundary testing
   - predict_raw1.py - End-to-end prediction pipeline

2. Financial Advisory System
   - threshold.py - Generate threshold rules
   - advisory_core2.py - Advisory logic

3. User Interface
   - gradio_ui.py - Main Gradio user interface
  
Execution Flow:
The system is designed to be executed in the following logical order.

1. Data exploration and preparation
   - Run eda.py to perform EDA and preprocess the data using pp3.py
2. Model development 
   - Train the models LR1.py and xgb3.py
3. Boundary testing
   - Run b1.py
4. Cross-domain testing
   - Run eval_my.py
5. Model validation
   - Run xgb_my.py
6. Prediction pipeline
   - Run predict_raw1.py
7. Generate the threshold rules
   - Run threshold.py
8. Generate financial advice, post-application guidance, and explanations
   - Run advisory_core2.py
9. Develop the user interface
   - Run gradio_ui.py
  
Dataset:
The priject uses a publicly available dataset obtained from Kaggle repository to build the model 
and uses a AI-generated Malaysia-based synthetic dataset to perform cross-domain testing and validate
the model's learnability under localized constraints and rules. 

Reproducibility:
All results can be reporduced by running the provided scripts and 
randomness is controlled where applicable, but minor variations in metrics are expected due to stochastic processes in machine learning.

Notes:
This repository demonstrates the methodology, system design, and implmentation. 









