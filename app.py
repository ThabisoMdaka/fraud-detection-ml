import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_all():
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
    return models, scaler_amount, \
           scaler_time, feature_names

models, scaler_amount, scaler_time, \
    feature_names = load_all()

# ===============================
# HEADER
# ===============================
st.markdown("""
<div style='background: linear-gradient(
    135deg, #1a237e, #283593);
    padding: 30px; border-radius: 12px;
    margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0;
               font-size: 2.2em;'>
        Credit Card Fraud Detection System
    </h1>
    <p style='color: #90CAF9; margin: 10px 0 0 0;
              font-size: 1.05em;'>
        Real-Time Transaction Risk Assessment |
        Logistic Regression · Random Forest · XGBoost
    </p>
    <p style='color: #90CAF9; margin: 5px 0 0 0;'>
        Built by <strong>Thabiso Mdaka</strong> |
        BSc Electronic Engineering | UKZN
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# METRICS ROW
# ===============================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Random Forest ROC-AUC", "0.9800")
with col2:
    st.metric("XGBoost Avg Precision", "0.8270")
with col3:
    st.metric("XGBoost False Alerts", "3")
with col4:
    st.metric("Training Transactions", "454,902")

st.markdown("---")

# ===============================
# SIDEBAR — TRANSACTION INPUT
# ===============================
st.sidebar.markdown("## Transaction Details")
st.sidebar.markdown("---")

st.sidebar.markdown("**Transaction Info**")
time_val   = st.sidebar.number_input(
    "Time (seconds from first transaction)",
    min_value=0, max_value=200000,
    value=50000, step=1000
)
amount_val = st.sidebar.number_input(
    "Transaction Amount ($)",
    min_value=0.0, max_value=30000.0,
    value=150.0, step=10.0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**PCA Features (V1 - V14)**"
)
st.sidebar.markdown(
    "*These are anonymised PCA-transformed "
    "features from the original transaction data.*"
)

v_vals = {}
col_a, col_b = st.sidebar.columns(2)
for i in range(1, 15):
    default = -3.0 if i in [1,3,7,9,10,12,14] \
              else 0.0
    if i % 2 == 1:
        v_vals[f'V{i}'] = col_a.number_input(
            f"V{i}", value=default,
            step=0.1, format="%.3f",
            key=f"v{i}"
        )
    else:
        v_vals[f'V{i}'] = col_b.number_input(
            f"V{i}", value=0.0,
            step=0.1, format="%.3f",
            key=f"v{i}"
        )

st.sidebar.markdown(
    "**PCA Features (V15 - V28)**"
)
col_c, col_d = st.sidebar.columns(2)
for i in range(15, 29):
    if i % 2 == 1:
        v_vals[f'V{i}'] = col_c.number_input(
            f"V{i}", value=0.0,
            step=0.1, format="%.3f",
            key=f"v{i}"
        )
    else:
        v_vals[f'V{i}'] = col_d.number_input(
            f"V{i}", value=0.0,
            step=0.1, format="%.3f",
            key=f"v{i}"
        )

st.sidebar.markdown("---")

# Quick load buttons
st.sidebar.markdown("**Quick Test Cases**")
col_quick1, col_quick2 = st.sidebar.columns(2)
load_fraud = col_quick1.button(
    "Load Fraud Case", use_container_width=True
)
load_legit = col_quick2.button(
    "Load Legit Case", use_container_width=True
)

assess_btn = st.sidebar.button(
    "Assess Transaction",
    use_container_width=True
)

# ===============================
# QUICK TEST CASES
# ===============================
fraud_case = {
    'V1': -3.0435, 'V2':  3.9256,
    'V3': -4.4779, 'V4':  4.1005,
    'V5': -3.4982, 'V6': -2.1296,
    'V7': -3.2194, 'V8':  2.6990,
    'V9': -3.7219, 'V10': -4.5669,
    'V11': 4.8704, 'V12': -5.0814,
    'V13': -0.2475,'V14': -9.4350,
    'V15': 0.6068, 'V16': -6.0489,
    'V17': -8.8319,'V18': -7.6305,
    'V19': 0.4036, 'V20':  0.5202,
    'V21': 0.6613, 'V22':  0.4352,
    'V23': -0.1828,'V24': -0.1453,
    'V25': 0.4127, 'V26': -0.0581,
    'V27': 0.8986, 'V28':  0.1703,
    'Time': 406, 'Amount': 77.89
}

legit_case = {
    'V1':  1.1918, 'V2':  0.2661,
    'V3':  0.1664, 'V4':  0.4481,
    'V5': -0.0438, 'V6': -0.1139,
    'V7':  0.0421, 'V8':  0.0597,
    'V9': -0.2730, 'V10': -0.1000,
    'V11': 0.0967, 'V12': -0.2063,
    'V13': 0.0658, 'V14':  0.0503,
    'V15': 0.2082, 'V16': -0.0989,
    'V17': -0.1220,'V18':  0.0849,
    'V19': -0.0426,'V20':  0.0214,
    'V21': -0.0123,'V22':  0.0201,
    'V23': 0.0149, 'V24': -0.0215,
    'V25': 0.0401, 'V26':  0.0172,
    'V27': 0.0058, 'V28':  0.0138,
    'Time': 86400, 'Amount': 42.50
}

if load_fraud:
    st.session_state['quick_case'] = fraud_case
    st.rerun()
if load_legit:
    st.session_state['quick_case'] = legit_case
    st.rerun()

# ===============================
# MAIN CONTENT
# ===============================
if assess_btn:
    transaction = {
        'Time':   time_val,
        'Amount': amount_val,
        **v_vals
    }

    # Preprocess
    df = pd.DataFrame([transaction])
    df['Amount_Scaled'] = \
        scaler_amount.transform(df[['Amount']])
    df['Time_Scaled'] = \
        scaler_time.transform(df[['Time']])
    df.drop(columns=['Amount', 'Time'],
            inplace=True)
    input_df = df[feature_names]

    # Predict
    predictions = {}
    for name, model in models.items():
        proba = float(
            model.predict_proba(input_df)[0][1]
        )
        predictions[name] = proba

    avg_proba   = np.mean(list(predictions.values()))
    fraud_votes = sum(
        1 for p in predictions.values() if p >= 0.5
    )
    is_fraud = fraud_votes >= 2

    # ===============================
    # RESULTS
    # ===============================
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.markdown("### Transaction Assessment")
        if is_fraud:
            st.markdown("""
<div style='background:#FFEBEE;
            border-left:6px solid #F44336;
            padding:20px; border-radius:8px;'>
    <h2 style='color:#C62828; margin:0;'>
        FRAUDULENT TRANSACTION
    </h2>
    <h3 style='color:#F44336; margin:8px 0;'>
        ACTION: BLOCK AND ALERT CUSTOMER
    </h3>
    <p style='color:#555; margin:5px 0;'>
        The majority of models have identified
        this transaction as fraudulent based on
        the PCA feature patterns associated with
        known fraud behaviour.
    </p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style='background:#E8F5E9;
            border-left:6px solid #4CAF50;
            padding:20px; border-radius:8px;'>
    <h2 style='color:#1B5E20; margin:0;'>
        LEGITIMATE TRANSACTION
    </h2>
    <h3 style='color:#4CAF50; margin:8px 0;'>
        ACTION: APPROVE AND PROCESS
    </h3>
    <p style='color:#555; margin:5px 0;'>
        The majority of models classify this
        transaction as legitimate. The feature
        profile is consistent with normal
        cardholder behaviour.
    </p>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Transaction Summary**")
        summary = pd.DataFrame({
            'Field': ['Time', 'Amount',
                      'V14 (Key Fraud Indicator)',
                      'V17 (Key Fraud Indicator)',
                      'Avg Fraud Probability',
                      'Model Agreement'],
            'Value': [
                f"{time_val:,} seconds",
                f"${amount_val:,.2f}",
                f"{v_vals['V14']:.4f}",
                f"{v_vals['V17']:.4f}",
                f"{avg_proba*100:.2f}%",
                f"{fraud_votes}/3 flag as fraud"
            ]
        })
        st.dataframe(summary,
                     use_container_width=True,
                     hide_index=True)

    with col_r2:
        st.markdown("### Fraud Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_proba * 100,
            domain={'x': [0,1], 'y': [0,1]},
            title={'text': "Average Fraud Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {
                    'color': "#F44336"
                    if is_fraud else "#4CAF50"
                },
                'steps': [
                    {'range': [0, 30],
                     'color': '#E8F5E9'},
                    {'range': [30, 60],
                     'color': '#FFF9C4'},
                    {'range': [60, 100],
                     'color': '#FFEBEE'}
                ],
                'threshold': {
                    'line': {'color': 'black',
                             'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(t=60, b=0, l=30, r=30)
        )
        st.plotly_chart(fig_gauge,
                        use_container_width=True)

    st.markdown("---")
    st.markdown("### Individual Model Predictions")

    cols = st.columns(3)
    colors_map = {
        'Logistic Regression': '#2196F3',
        'Random Forest':       '#4CAF50',
        'XGBoost':             '#FF9800'
    }
    for col, (name, proba) in zip(
            cols, predictions.items()
    ):
        with col:
            decision = "FRAUDULENT" \
                if proba >= 0.5 \
                else "LEGITIMATE"
            border_color = "#F44336" \
                if proba >= 0.5 \
                else "#4CAF50"
            st.markdown(f"""
<div style='border:2px solid {border_color};
            border-radius:10px;
            padding:15px;
            text-align:center;'>
    <h4 style='color:#333; margin:0 0 8px 0;'>
        {name}
    </h4>
    <h2 style='color:{border_color}; margin:0;'>
        {proba*100:.2f}%
    </h2>
    <p style='color:#666; margin:5px 0 0 0;
              font-weight:bold;'>
        {decision}
    </p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Bar chart comparison
    fig_bar = px.bar(
        x=list(predictions.keys()),
        y=[v*100 for v in predictions.values()],
        color=[v*100 for v in predictions.values()],
        color_continuous_scale=[
            '#4CAF50', '#FFC107', '#F44336'
        ],
        labels={
            'x': 'Model',
            'y': 'Fraud Probability (%)'
        },
        title='Fraud Probability — Model Comparison'
    )
    fig_bar.add_hline(
        y=50, line_dash="dash",
        line_color="black",
        annotation_text="Decision Threshold (50%)"
    )
    fig_bar.update_layout(
        showlegend=False, height=400
    )
    st.plotly_chart(fig_bar,
                    use_container_width=True)

else:
    # Welcome screen
    st.markdown("### Enter transaction details in the sidebar and click Assess Transaction")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div style='background:#E3F2FD;
            border-radius:10px;
            padding:20px; text-align:center;'>
    <h3>3 AI Models</h3>
    <p>Logistic Regression, Random Forest
    and XGBoost with majority vote</p>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div style='background:#E8F5E9;
            border-radius:10px;
            padding:20px; text-align:center;'>
    <h3>98% ROC-AUC</h3>
    <p>Random Forest achieves
    industry-grade discrimination</p>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div style='background:#FFF3E0;
            border-radius:10px;
            padding:20px; text-align:center;'>
    <h3>284,807 Transactions</h3>
    <p>Trained on real anonymised
    European credit card data</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
### About This System

This fraud detection system was built to
demonstrate applied machine learning skills
in a financial services context.

The system addresses one of the most
challenging problems in financial data science:
detecting rare fraudulent events (0.173% of
transactions) within a massively imbalanced
dataset of 284,807 real credit card transactions.

**Key technical challenges solved:**
- Class imbalance handled using SMOTE oversampling
- Optimal decision threshold found per model
- Business impact quantified in dollar terms
- Three models compared with majority vote ensemble

**Performance highlights:**
- Random Forest: ROC-AUC = 0.9800
- XGBoost: Average Precision = 0.8270
- XGBoost at optimal threshold: only 3 false alerts
  while catching 77.6% of all fraud cases
    """)