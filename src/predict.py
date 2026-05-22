import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===============================
# FRAUD DETECTION PREDICTION PIPELINE
# Author: Thabiso Mdaka | UKZN
# ===============================

def load_models():
    """Load all saved models and preprocessing objects"""
    models = {
        'Logistic Regression': joblib.load(
            'models/logistic_regression.pkl'
        ),
        'Random Forest': joblib.load(
            'models/random_forest.pkl'
        ),
        'XGBoost': joblib.load(
            'models/xgboost.pkl'
        )
    }
    scaler_amount = joblib.load(
        'models/scaler_amount.pkl'
    )
    scaler_time = joblib.load(
        'models/scaler_time.pkl'
    )
    feature_names = joblib.load(
        'models/feature_names.pkl'
    )
    return models, scaler_amount, scaler_time, feature_names


def preprocess_transaction(transaction: dict,
                           scaler_amount,
                           scaler_time,
                           feature_names: list):
    """
    Preprocess a raw transaction for model input.

    Parameters:
    -----------
    transaction : dict
        Raw transaction with keys:
        Time, V1-V28, Amount
    scaler_amount : StandardScaler
        Fitted scaler for Amount feature
    scaler_time : StandardScaler
        Fitted scaler for Time feature
    feature_names : list
        Ordered list of model input features

    Returns:
    --------
    input_df : DataFrame
        Preprocessed transaction ready for prediction
    """
    df = pd.DataFrame([transaction])

    df['Amount_Scaled'] = scaler_amount.transform(
        df[['Amount']]
    )
    df['Time_Scaled'] = scaler_time.transform(
        df[['Time']]
    )
    df.drop(columns=['Amount', 'Time'], inplace=True)

    input_df = df[feature_names]
    return input_df


def predict_transaction(transaction: dict,
                        threshold: float = 0.5):
    """
    Predict whether a transaction is fraudulent.

    Parameters:
    -----------
    transaction : dict
        Raw transaction features
    threshold : float
        Decision threshold (default 0.5)

    Returns:
    --------
    results : dict
        Predictions from all models
    """
    models, scaler_amount, scaler_time, \
        feature_names = load_models()

    input_df = preprocess_transaction(
        transaction, scaler_amount,
        scaler_time, feature_names
    )

    results = {}
    for name, model in models.items():
        proba = float(
            model.predict_proba(input_df)[0][1]
        )
        decision = 'FRAUDULENT — BLOCK' \
            if proba >= threshold \
            else 'LEGITIMATE — APPROVE'
        confidence = proba * 100 \
            if proba >= threshold \
            else (1 - proba) * 100

        results[name] = {
            'fraud_probability': round(proba * 100, 3),
            'decision':          decision,
            'confidence':        round(confidence, 3)
        }

    return results


def print_assessment(transaction: dict,
                     results: dict,
                     threshold: float = 0.5):
    """Print formatted fraud assessment report"""
    print("\n" + "="*60)
    print("   FRAUD TRANSACTION ASSESSMENT REPORT")
    print("="*60)
    print(f"\nTransaction Details:")
    print(f"  Time:   {transaction['Time']:,.0f} seconds")
    print(f"  Amount: ${transaction['Amount']:,.2f}")
    print(f"  V1:     {transaction['V1']:.4f}")
    print(f"  V14:    {transaction['V14']:.4f}")
    print(f"  V17:    {transaction['V17']:.4f}")

    print("\n" + "-"*60)
    print("Model Predictions:")
    print("-"*60)

    for name, result in results.items():
        status = "FRAUD" \
            if "FRAUDULENT" in result['decision'] \
            else "LEGITIMATE"
        print(f"\n  {name}:")
        print(f"    Fraud Probability: "
              f"{result['fraud_probability']:.3f}%")
        print(f"    Decision:          "
              f"{result['decision']}")
        print(f"    Confidence:        "
              f"{result['confidence']:.3f}%")

    print("\n" + "-"*60)
    probas = [r['fraud_probability']
              for r in results.values()]
    avg_proba = np.mean(probas)
    fraud_votes = sum(
        1 for r in results.values()
        if "FRAUDULENT" in r['decision']
    )

    print("FINAL ASSESSMENT (Majority Vote):")
    if fraud_votes >= 2:
        print(f"  STATUS:      FRAUDULENT TRANSACTION")
        print(f"  ACTION:      BLOCK AND ALERT CUSTOMER")
        print(f"  Avg Probability: {avg_proba:.3f}%")
        print(f"  Model Agreement: "
              f"{fraud_votes}/3 models flag as fraud")
    else:
        print(f"  STATUS:      LEGITIMATE TRANSACTION")
        print(f"  ACTION:      APPROVE AND PROCESS")
        print(f"  Avg Probability: {avg_proba:.3f}%")
        print(f"  Model Agreement: "
              f"{3-fraud_votes}/3 models flag as legitimate")
    print("="*60)


# ===============================
# TEST CASES
# ===============================
if __name__ == "__main__":

    print("=" * 60)
    print("   FRAUD DETECTION PREDICTION PIPELINE")
    print("   Author: Thabiso Mdaka | UKZN")
    print("=" * 60)

    # Test 1: Known fraudulent transaction
    # (high V14 negative value is a fraud indicator)
    fraud_transaction = {
        'Time': 406,
        'V1':  -3.0435, 'V2':   3.9256,
        'V3':  -4.4779, 'V4':   4.1005,
        'V5':  -3.4982, 'V6':  -2.1296,
        'V7':  -3.2194, 'V8':   2.6990,
        'V9':  -3.7219, 'V10': -4.5669,
        'V11':  4.8704, 'V12': -5.0814,
        'V13': -0.2475, 'V14': -9.4350,
        'V15':  0.6068, 'V16': -6.0489,
        'V17': -8.8319, 'V18': -7.6305,
        'V19':  0.4036, 'V20':  0.5202,
        'V21':  0.6613, 'V22':  0.4352,
        'V23': -0.1828, 'V24': -0.1453,
        'V25':  0.4127, 'V26': -0.0581,
        'V27':  0.8986, 'V28':  0.1703,
        'Amount': 77.89
    }

    print("\n TEST 1 — Expected: FRAUDULENT")
    results1 = predict_transaction(
        fraud_transaction, threshold=0.5
    )
    print_assessment(fraud_transaction, results1)

    # Test 2: Typical legitimate transaction
    legit_transaction = {
        'Time': 86400,
        'V1':   1.1918, 'V2':   0.2661,
        'V3':   0.1664, 'V4':   0.4481,
        'V5':  -0.0438, 'V6':  -0.1139,
        'V7':   0.0421, 'V8':   0.0597,
        'V9':  -0.2730, 'V10': -0.1000,
        'V11':  0.0967, 'V12': -0.2063,
        'V13':  0.0658, 'V14':  0.0503,
        'V15':  0.2082, 'V16': -0.0989,
        'V17': -0.1220, 'V18':  0.0849,
        'V19': -0.0426, 'V20':  0.0214,
        'V21': -0.0123, 'V22':  0.0201,
        'V23':  0.0149, 'V24': -0.0215,
        'V25':  0.0401, 'V26':  0.0172,
        'V27':  0.0058, 'V28':  0.0138,
        'Amount': 42.50
    }

    print("\n TEST 2 — Expected: LEGITIMATE")
    results2 = predict_transaction(
        legit_transaction, threshold=0.5
    )
    print_assessment(legit_transaction, results2)

    # Test 3: High value borderline transaction
    borderline_transaction = {
        'Time': 50000,
        'V1':  -1.3598, 'V2':  -0.0728,
        'V3':   2.5363, 'V4':   1.3782,
        'V5':  -0.3383, 'V6':   0.4624,
        'V7':   0.2396, 'V8':   0.0987,
        'V9':   0.3638, 'V10':  0.0908,
        'V11': -0.5516, 'V12': -0.6178,
        'V13': -0.9913, 'V14': -0.3112,
        'V15':  1.4681, 'V16': -0.4704,
        'V17':  0.2076, 'V18':  0.0258,
        'V19':  0.4039, 'V20':  0.2514,
        'V21': -0.0183, 'V22':  0.2778,
        'V23': -0.1104, 'V24':  0.0669,
        'V25':  0.1285, 'V26': -0.1891,
        'V27':  0.1336, 'V28': -0.0210,
        'Amount': 4500.00
    }

    print("\n TEST 3 — High Value Transaction")
    results3 = predict_transaction(
        borderline_transaction, threshold=0.5
    )
    print_assessment(borderline_transaction, results3)