# UPI Payment Failure Predictor

Built for the **Razorpay AI Buildathon 2026** — Open Track.

## Problem Statement

Payment failures on UPI create friction for merchants and customers alike. This project builds a machine learning model that predicts whether a UPI transaction will **SUCCEED** or **FAIL**, based on transaction amount, time of day, day of week, and the sender/receiver bank pair — enabling merchants to proactively flag or retry risky transactions.

## Dataset

- Simulated UPI transaction dataset (1,000 records)
- Fields: Transaction ID, Timestamp, Sender/Receiver Name, Sender/Receiver UPI ID, Amount (INR), Status
- Target: `Status` (SUCCESS / FAILED) — near-balanced classes (502 vs 498)

## Approach

1. **Feature Engineering**
   - Extracted `hour` and `day_of_week` from timestamp
   - Extracted `sender_bank` / `receiver_bank` from UPI ID suffixes (e.g., `@okhdfcbank`)
   - Engineered a `same_bank` flag (same-bank transactions vs cross-bank)

2. **Preprocessing**
   - One-hot encoded categorical bank features
   - 80/20 stratified train/test split

3. **Modeling**
   - Trained two models for comparison: **Logistic Regression** (baseline) and **Random Forest**
   - Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix (not accuracy alone, since false negatives on failures are costly)

4. **Edge Case Testing**
   - Stress-tested the model with an unseen/unknown bank scenario
   - **Finding:** the model's confidence collapses to near-50/50 (51.5%) when it encounters a bank not seen during training — a known limitation of one-hot encoding on a small dataset
   - **Mitigation:** in production, this would be handled with a fallback "unknown_bank" category during encoding, or periodic retraining as new banks appear

5. **Demo**
   - Built an interactive Streamlit app where a user inputs transaction details and gets a live SUCCESS/FAILED prediction with confidence scores

## Results

*(Fill in your actual Step 5 numbers here, e.g.:)*
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |

## How to Run

```bash
# Clone the repo
git clone https://github.com/kishorereddycse860-hub/razorpay-upi-failure-predictor.git
cd razorpay-upi-failure-predictor

# Install dependencies
pip install -r requirements.txt

# Run the scripts in order (steps 1-6)
python scripts/step1_load_data.py
python scripts/step2_feature_engineering.py
python scripts/step3_encode_and_split.py
python scripts/step4_model_building.py
python scripts/step5_evaluation.py
python scripts/step6_edge_case_testing.py

# Launch the demo app
streamlit run app.py
```

## Tech Stack

Python, Pandas, Scikit-learn, Streamlit, Joblib

## Author

Kishore Reddy Gayam — B.Tech CSE (AI/ML), Marwadi University
