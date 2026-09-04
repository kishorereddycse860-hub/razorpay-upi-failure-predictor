# UPI Payment Failure Predictor

Built for the Razorpay AI Buildathon 2026 — Open Track.

## Problem Statement

Payment failures on UPI create friction for merchants and customers alike. This project builds a machine learning model that predicts whether a UPI transaction will SUCCEED or FAIL, based on transaction amount, time of day, day of week, and the sender/receiver bank pair — enabling merchants to proactively flag or retry risky transactions.

## Dataset

- Simulated UPI transaction dataset (1,000 records)
- Fields: Transaction ID, Timestamp, Sender/Receiver Name, Sender/Receiver UPI ID, Amount (INR), Status
- Target: Status (SUCCESS / FAILED) — near-balanced classes (502 vs 498)

## Approach

### Feature Engineering
- Extracted `hour` and `day_of_week` from timestamp
- Extracted `sender_bank` / `receiver_bank` from UPI ID suffixes (e.g., `@okhdfcbank`)
- Engineered a `same_bank` flag (same-bank transactions vs cross-bank)

### Preprocessing
- One-hot encoded categorical bank features
- 80/20 stratified train/test split

### Modeling
- Trained three models for comparison: Logistic Regression (baseline), Random Forest, and XGBoost
- Ran hyperparameter tuning on Random Forest using GridSearchCV (5-fold cross-validation, 27 parameter combinations, 135 total fits)
- Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix (not accuracy alone, since false negatives on failures are costly)

### Edge Case Testing
- Stress-tested the model with an unseen/unknown bank scenario
- Finding: the model's confidence collapses to near-50/50 (51.5%) when it encounters a bank not seen during training — a known limitation of one-hot encoding on a small dataset
- Mitigation: in production, this would be handled with a fallback "unknown_bank" category during encoding, or periodic retraining as new banks appear

### Demo
- Built an interactive Streamlit app where a user inputs transaction details and gets a live SUCCESS/FAILED prediction with confidence scores

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.53 | 0.53 | 0.55 | 0.54 |
| Random Forest | 0.47 | 0.47 | 0.49 | 0.48 |
| Random Forest (Tuned) | 0.50 | 0.50 | 0.55 | 0.52 |

## Key Finding

- All three models — Logistic Regression, Random Forest, and XGBoost — converge to chance-level accuracy (47–53%) on this balanced binary classification task
- This held even after extensive hyperparameter tuning (135 model fits via GridSearchCV)
- This is not a pipeline bug — the pipeline was verified at every stage (feature engineering, encoding, splitting, training, evaluation)
- The conclusion: transaction metadata alone (amount, timing, bank identity) does not carry meaningful predictive signal for UPI failure in this dataset

## Interpretation

- Real-world UPI payment failures are more likely driven by infrastructure-level factors not captured in this dataset — network conditions, bank-side processing delays, server timeouts, or gateway issues
- A production-grade failure predictor would need richer, real-time signals (e.g., live network latency, bank API response times, historical failure rates per bank pair) rather than static transaction attributes
- This finding is itself valuable: it tells a payments team where *not* to invest modeling effort, and points toward what data would actually need to be collected

## How to Run

```bash
# Clone the repo
git clone https://github.com/kishorereddycse860-hub/razorpay-upi-failure-predictor.git
cd razorpay-upi-failure-predictor

# Install dependencies
pip install -r requirements.txt

# Run the notebook (RAZAR_PAY.ipynb) to reproduce feature engineering,
# training, evaluation, edge-case testing, and tuning

# Launch the demo app
streamlit run app.py
```

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Streamlit, Joblib

## Architecture

Raw UPI Transaction Dataset (CSV)
│
▼
Feature Engineering

Extract hour, day of week from timestamp
Extract bank from UPI ID
same_bank flag
│
▼
Encoding + Train/Test Split
One-hot encode bank features
80/20 stratified split
│
▼
Model Training
Logistic Regression (baseline)
Random Forest
XGBoost
Hyperparameter tuning (GridSearchCV, 5-fold CV)
│
▼
Evaluation
Accuracy, Precision, Recall, F1, Confusion Matrix
Edge case testing (unseen bank scenario)
│
▼
Streamlit Demo App
User inputs transaction details
Live SUCCESS/FAILED prediction with confidence score

## Author

Kishore Reddy Gayam — B.Tech CSE (AI/ML), Marwadi University
