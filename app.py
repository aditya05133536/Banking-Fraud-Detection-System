import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# Configuration
MODEL_PATH = 'model.joblib'
ANOMALY_MODEL_PATH = 'iso_forest.joblib'
FEATURES = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
            'errorBalanceOrig','errorBalanceDest'] + [f'type_{t}' for t in
            ['CASH_OUT','CASH_IN','DEBIT','PAYMENT','TRANSFER']]

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    return joblib.load(path)

def preprocess(df):
    df = df.copy()
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    types = ['CASH_OUT','CASH_IN','DEBIT','PAYMENT','TRANSFER']
    for t in types:
        df[f'type_{t}'] = (df['type'] == t).astype(int)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0
    return df[FEATURES]

def risk_tier(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.3:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def main():
    st.title('Banking Fraud Detection System')

    uploaded_file = st.file_uploader('Upload transactions CSV', type=['csv'])
    if uploaded_file is None:
        st.info('Please upload a transaction CSV file.')
        return

    df = pd.read_csv(uploaded_file)

    try:
        model = load_model(MODEL_PATH)
        anomaly_model = load_model(ANOMALY_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    X = preprocess(df)

    # Align columns with model
    model_features = model.feature_names_in_
    X = X[model_features]

    # Predict with supervised model
    fraud_probs = model.predict_proba(X)[:, 1]

    # Predict with anomaly detector
    anomaly_scores = -anomaly_model.decision_function(X).reshape(-1, 1)
    anomaly_probs = MinMaxScaler().fit_transform(anomaly_scores).flatten()

    # Ensemble
    clf_weight = st.sidebar.slider('Classifier Weight', 0.0, 1.0, 0.7, 0.05)
    combined_probs = clf_weight * fraud_probs + (1 - clf_weight) * anomaly_probs

    threshold = st.sidebar.slider('Fraud Probability Threshold', 0.0, 1.0, 0.5, 0.01)
    preds = (combined_probs >= threshold).astype(int)

    df['fraud_probability'] = combined_probs
    df['fraud_prediction'] = preds
    df['risk_tier'] = df['fraud_probability'].apply(risk_tier)

    # Summary statistics
    st.header("Prediction Summary")
    st.write(f"Total transactions: {len(df)}")
    st.write(f"Predicted fraud transactions: {preds.sum()}")
    st.write(f"Predicted normal transactions: {len(df) - preds.sum()}")

    st.header("Risk Tier Counts")
    st.write(df['risk_tier'].value_counts())

    # Show top fraud risks
    st.header("Top Suspicious Transactions")
    st.dataframe(df.sort_values('fraud_probability', ascending=False).head(20))

    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "fraud_predictions.csv")

if __name__ == "__main__":
    main()
