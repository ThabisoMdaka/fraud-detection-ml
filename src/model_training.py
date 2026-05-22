import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

print("=" * 60)
print("   FRAUD DETECTION — Model Training Pipeline")
print("=" * 60)

# ===============================
# 1. LOAD PROCESSED DATA
# ===============================
print("\n--- Loading Processed Data ---")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test  = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"Training: {len(X_train):,} samples")
print(f"Test:     {len(X_test):,} samples")

# ===============================
# 2. SMOTE OVERSAMPLING
# ===============================
print("\n--- Applying SMOTE Oversampling ---")
print("SMOTE generates synthetic minority class samples")
print("by interpolating between existing fraud cases.")
print("Applied ONLY to training data — never test data.")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(
    X_train, y_train
)

print(f"\nBefore SMOTE: {y_train.sum():,} fraud samples")
print(f"After SMOTE:  {y_train_sm.sum():,} fraud samples")
print(f"After SMOTE:  {(y_train_sm==0).sum():,} legitimate samples")
print(f"Total training samples: {len(X_train_sm):,}")

# ===============================
# 3. TRAIN LOGISTIC REGRESSION
# ===============================
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train_sm, y_train_sm)
lr_pred  = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_ap    = average_precision_score(y_test, lr_proba)
print(f"ROC-AUC:           {lr_auc:.4f}")
print(f"Average Precision: {lr_ap:.4f}")

# ===============================
# 4. TRAIN RANDOM FOREST
# ===============================
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_sm, y_train_sm)
rf_pred  = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
rf_ap    = average_precision_score(y_test, rf_proba)
print(f"ROC-AUC:           {rf_auc:.4f}")
print(f"Average Precision: {rf_ap:.4f}")

# ===============================
# 5. TRAIN XGBOOST
# ===============================
print("\n--- Training XGBoost ---")
scale_pos_weight = (y_train_sm==0).sum() / y_train_sm.sum()
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    scale_pos_weight=scale_pos_weight
)
xgb_model.fit(X_train_sm, y_train_sm)
xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_ap    = average_precision_score(y_test, xgb_proba)
print(f"ROC-AUC:           {xgb_auc:.4f}")
print(f"Average Precision: {xgb_ap:.4f}")

# ===============================
# 6. SAVE ALL MODELS
# ===============================
print("\n--- Saving Models ---")
joblib.dump(lr_model,  'models/logistic_regression.pkl')
joblib.dump(rf_model,  'models/random_forest.pkl')
joblib.dump(xgb_model, 'models/xgboost.pkl')
print("Saved: models/logistic_regression.pkl")
print("Saved: models/random_forest.pkl")
print("Saved: models/xgboost.pkl")

# ===============================
# 7. VISUALIZATIONS
# ===============================
print("\n--- Generating Result Plots ---")

# Plot 1: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, pred, title in zip(
        axes,
        [lr_pred, rf_pred, xgb_pred],
        ['Logistic Regression',
         'Random Forest', 'XGBoost']):
    cm = confusion_matrix(y_test, pred)
    cm_norm = cm.astype('float') / \
              cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt='.3f',
        cmap='Blues',
        xticklabels=['Legitimate', 'Fraudulent'],
        yticklabels=['Legitimate', 'Fraudulent'],
        ax=ax, linewidths=0.5
    )
    ax.set_title(f'{title}\nConfusion Matrix',
                 fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle(
    'Normalised Confusion Matrices — All Models\n'
    'Test Set: 56,962 transactions | 98 fraudulent',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/confusion_matrices.png")

# Plot 2: ROC Curves
plt.figure(figsize=(10, 7))
for proba, label, color in zip(
        [lr_proba, rf_proba, xgb_proba],
        ['Logistic Regression',
         'Random Forest', 'XGBoost'],
        ['#2196F3', '#4CAF50', '#F44336']):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2.5,
             label=f'{label} (AUC = {auc:.4f})',
             color=color)
plt.plot([0,1],[0,1], 'k--',
         linewidth=1.5,
         label='Random Classifier (AUC = 0.5000)')
plt.fill_between(fpr, tpr, alpha=0.05,
                 color='#2196F3')
plt.title(
    'ROC Curve Comparison — Fraud Detection Models\n'
    'Measuring ability to rank fraudulent transactions '
    'above legitimate ones',
    fontsize=12, fontweight='bold'
)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/roc_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/roc_curves.png")

# Plot 3: Precision-Recall Curves
plt.figure(figsize=(10, 7))
for proba, label, color in zip(
        [lr_proba, rf_proba, xgb_proba],
        ['Logistic Regression',
         'Random Forest', 'XGBoost'],
        ['#2196F3', '#4CAF50', '#F44336']):
    precision, recall, _ = \
        precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.plot(recall, precision, linewidth=2.5,
             label=f'{label} (AP = {ap:.4f})',
             color=color)
baseline = y_test.mean()
plt.axhline(y=baseline, color='gray',
            linestyle='--', linewidth=1.5,
            label=f'Baseline (AP = {baseline:.4f})')
plt.title(
    'Precision-Recall Curve — Fraud Detection Models\n'
    'Primary metric for severely imbalanced datasets',
    fontsize=12, fontweight='bold'
)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/precision_recall.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/precision_recall.png")

# Plot 4: Feature Importance (Random Forest)
feat_imp_rf = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True).tail(15)

plt.figure(figsize=(12, 8))
colors = plt.cm.RdYlGn(
    np.linspace(0.2, 0.9, len(feat_imp_rf))
)
plt.barh(feat_imp_rf.index, feat_imp_rf.values,
         color=colors, edgecolor='black',
         linewidth=0.5)
plt.title(
    'Top 15 Most Important Features — Random Forest\n'
    'Variables most predictive of fraudulent transactions',
    fontsize=12, fontweight='bold'
)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance_rf.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/feature_importance_rf.png")

# Plot 5: Feature Importance (XGBoost)
feat_imp_xgb = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=True).tail(15)

plt.figure(figsize=(12, 8))
colors = plt.cm.Blues(
    np.linspace(0.3, 0.9, len(feat_imp_xgb))
)
plt.barh(feat_imp_xgb.index, feat_imp_xgb.values,
         color=colors, edgecolor='black',
         linewidth=0.5)
plt.title(
    'Top 15 Most Important Features — XGBoost\n'
    'Variables most predictive of fraudulent transactions',
    fontsize=12, fontweight='bold'
)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance_xgb.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/feature_importance_xgb.png")

# Plot 6: Score Distribution
plt.figure(figsize=(12, 6))
plt.hist(rf_proba[y_test==0], bins=50,
         alpha=0.7, color='#2196F3',
         label='Legitimate Transactions',
         density=True)
plt.hist(rf_proba[y_test==1], bins=50,
         alpha=0.7, color='#F44336',
         label='Fraudulent Transactions',
         density=True)
plt.axvline(x=0.5, color='black',
            linestyle='--', linewidth=2,
            label='Decision Threshold (0.5)')
plt.title(
    'Predicted Fraud Probability Distribution\n'
    'Separation of Legitimate vs Fraudulent Transactions '
    '(Random Forest)',
    fontsize=12, fontweight='bold'
)
plt.xlabel('Predicted Fraud Probability', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/score_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/score_distribution.png")

# ===============================
# 8. SAVE EVALUATION REPORT
# ===============================
report = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest',
        'XGBoost'
    ],
    'ROC_AUC': [
        round(lr_auc, 4),
        round(rf_auc, 4),
        round(xgb_auc, 4)
    ],
    'Average_Precision': [
        round(lr_ap, 4),
        round(rf_ap, 4),
        round(xgb_ap, 4)
    ]
})
report.to_csv(
    'outputs/reports/model_comparison.csv',
    index=False
)

# ===============================
# 9. FINAL REPORT
# ===============================
print("\n" + "="*60)
print("   FINAL EVALUATION REPORT")
print("="*60)

print("\nLOGISTIC REGRESSION:")
print(classification_report(
    y_test, lr_pred,
    target_names=['Legitimate', 'Fraudulent']
))

print("RANDOM FOREST:")
print(classification_report(
    y_test, rf_pred,
    target_names=['Legitimate', 'Fraudulent']
))

print("XGBOOST:")
print(classification_report(
    y_test, xgb_pred,
    target_names=['Legitimate', 'Fraudulent']
))

print("MODEL COMPARISON:")
print(f"{'Model':<25} {'ROC-AUC':>8} {'Avg Precision':>15}")
print("-"*50)
for _, row in report.iterrows():
    print(f"{row['Model']:<25} "
          f"{row['ROC_AUC']:>8.4f} "
          f"{row['Average_Precision']:>15.4f}")

print("\nAll plots saved to outputs/plots/")
print("Report saved to outputs/reports/")
print("="*60)