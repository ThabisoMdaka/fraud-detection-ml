# ===============================
#  Fraud Detection ML System
#  Author: Thabiso Mdaka
#  Dataset: Credit Card Fraud Detection (Kaggle)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

# ===============================
# 1. LOAD & EXPLORE DATA
# ===============================
print("=" * 55)
print("   FRAUD DETECTION ML SYSTEM — Thabiso Mdaka")
print("=" * 55)
print("\n--- Loading Dataset ---")

df = pd.read_csv('creditcard.csv')

print(f"Dataset Shape: {df.shape}")
print(f"Total Transactions: {len(df):,}")
print(f"Fraudulent: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print(f"Legitimate: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.3f}%)")

# ===============================
# 2. VISUALIZATIONS — DATA UNDERSTANDING
# ===============================
print("\n--- Generating Data Analysis Plots ---")

# Plot 1: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
counts = df['Class'].value_counts()
bars = axes[0].bar(['Legitimate', 'Fraudulent'],
                   counts.values,
                   color=['#2196F3', '#F44336'],
                   edgecolor='black', linewidth=0.8)
for bar, count in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 500,
                 f'{count:,}', ha='center',
                 fontweight='bold', fontsize=11)
axes[0].set_title('Class Distribution\n(Severe Imbalance)', fontweight='bold')
axes[0].set_ylabel('Number of Transactions')
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart
axes[1].pie(counts.values,
            labels=['Legitimate\n(99.83%)', 'Fraudulent\n(0.17%)'],
            colors=['#2196F3', '#F44336'],
            explode=(0, 0.1),
            autopct='%1.2f%%',
            startangle=90)
axes[1].set_title('Transaction Distribution\n(Pie Chart)', fontweight='bold')

plt.suptitle('Credit Card Fraud Dataset — Class Imbalance Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/class_distribution.png', dpi=150)
plt.close()
print("Saved: results/class_distribution.png")

# Plot 2: Transaction Amount Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fraud = df[df['Class'] == 1]['Amount']
legit = df[df['Class'] == 0]['Amount']

axes[0].hist(legit, bins=50, color='#2196F3',
             alpha=0.7, label='Legitimate', density=True)
axes[0].hist(fraud, bins=50, color='#F44336',
             alpha=0.7, label='Fraudulent', density=True)
axes[0].set_title('Transaction Amount Distribution', fontweight='bold')
axes[0].set_xlabel('Amount (£)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim([0, 1000])
axes[0].grid(True, alpha=0.3)

axes[1].boxplot([legit.clip(upper=500), fraud.clip(upper=500)],
                labels=['Legitimate', 'Fraudulent'],
                patch_artist=True,
                boxprops=dict(facecolor='#2196F3', alpha=0.7))
axes[1].set_title('Transaction Amount Boxplot\n(clipped at £500)',
                  fontweight='bold')
axes[1].set_ylabel('Amount (£)')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Transaction Amount Analysis — Fraud vs Legitimate',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/amount_distribution.png', dpi=150)
plt.close()
print("Saved: results/amount_distribution.png")

# ===============================
# 3. PREPROCESSING
# ===============================
print("\n--- Preprocessing Data ---")

# Scale Amount and Time
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Train/Test Split BEFORE SMOTE (critical!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")
print(f"Fraud in test set: {y_test.sum()} samples")

# Apply SMOTE only on training data
print("\n--- Applying SMOTE to balance training data ---")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Training samples: {len(X_train_sm):,}")
print(f"Fraud samples: {y_train_sm.sum():,}")
print(f"Legitimate samples: {(y_train_sm==0).sum():,}")

# ===============================
# 4. TRAIN RANDOM FOREST
# ===============================
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
print(f" Random Forest ROC-AUC: {rf_auc:.4f}")

# ===============================
# 5. TRAIN XGBOOST
# ===============================
print("\n--- Training XGBoost ---")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train_sm, y_train_sm)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f" XGBoost ROC-AUC: {xgb_auc:.4f}")

# ===============================
# 6. RESULTS & VISUALIZATIONS
# ===============================
print("\n--- Generating Result Plots ---")

# Plot 3: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pred, title in zip(axes,
                            [rf_pred, xgb_pred],
                            ['Random Forest', 'XGBoost']):
    cm = confusion_matrix(y_test, pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'],
                ax=ax)
    ax.set_title(f'{title} — Confusion Matrix', fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Model Comparison — Confusion Matrices',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=150)
plt.close()
print(" Saved: results/confusion_matrices.png")

# Plot 4: ROC Curves
plt.figure(figsize=(10, 7))
for proba, label, color in zip(
        [rf_proba, xgb_proba],
        ['Random Forest', 'XGBoost'],
        ['#2196F3', '#F44336']):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2.5,
             label=f'{label} (AUC = {auc:.4f})',
             color=color)

plt.plot([0, 1], [0, 1], 'k--',
         linewidth=1.5, label='Random Classifier (AUC = 0.5)')
plt.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
plt.title('ROC Curve Comparison — Fraud Detection Models',
          fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=150)
plt.close()
print(" Saved: results/roc_curves.png")

# Plot 5: Feature Importance (XGBoost)
plt.figure(figsize=(12, 8))
feat_importance = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True).tail(15)

colors = plt.cm.Blues(
    np.linspace(0.4, 1.0, len(feat_importance))
)
bars = plt.barh(feat_importance.index,
                feat_importance.values,
                color=colors, edgecolor='black',
                linewidth=0.5)
plt.title('Top 15 Most Important Features — XGBoost\n'
          '(Features most predictive of fraud)',
          fontsize=13, fontweight='bold')
plt.xlabel('Feature Importance Score', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150)
plt.close()
print(" Saved: results/feature_importance.png")

# Plot 6: Precision-Recall Curve
plt.figure(figsize=(10, 7))
for proba, label, color in zip(
        [rf_proba, xgb_proba],
        ['Random Forest', 'XGBoost'],
        ['#2196F3', '#F44336']):
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.plot(recall, precision, linewidth=2.5,
             label=f'{label} (AP = {ap:.4f})',
             color=color)

plt.title('Precision-Recall Curve — Fraud Detection\n'
          '(Critical metric for imbalanced datasets)',
          fontsize=13, fontweight='bold')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/precision_recall.png', dpi=150)
plt.close()
print(" Saved: results/precision_recall.png")

# ===============================
# 7. FINAL REPORT
# ===============================
print("\n" + "="*55)
print("   FINAL EVALUATION REPORT")
print("="*55)

print("\n RANDOM FOREST:")
print(classification_report(y_test, rf_pred,
      target_names=['Legitimate', 'Fraudulent']))

print("\n XGBOOST:")
print(classification_report(y_test, xgb_pred,
      target_names=['Legitimate', 'Fraudulent']))

print("\n MODEL COMPARISON:")
print(f"{'Model':<20} {'ROC-AUC':<12} {'AP Score'}")
print("-"*45)
print(f"{'Random Forest':<20} "
      f"{rf_auc:<12.4f} "
      f"{average_precision_score(y_test, rf_proba):.4f}")
print(f"{'XGBoost':<20} "
      f"{xgb_auc:<12.4f} "
      f"{average_precision_score(y_test, xgb_proba):.4f}")

print("\n All results saved to results/ folder!")
print("="*55)
