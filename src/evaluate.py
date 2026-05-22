import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    precision_recall_curve
)
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   FRAUD DETECTION — Model Evaluation")
print("=" * 60)

# ===============================
# 1. LOAD DATA AND MODELS
# ===============================
X_test  = pd.read_csv('data/processed/X_test.csv')
y_test  = pd.read_csv('data/processed/y_test.csv').squeeze()

lr_model  = joblib.load('models/logistic_regression.pkl')
rf_model  = joblib.load('models/random_forest.pkl')
xgb_model = joblib.load('models/xgboost.pkl')

lr_proba  = lr_model.predict_proba(X_test)[:, 1]
rf_proba  = rf_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# ===============================
# 2. THRESHOLD ANALYSIS
# ===============================
print("\n--- Threshold Analysis ---")
print("Finding optimal decision threshold for each model")
print("Optimising for F1-Score on the fraud class\n")

def find_optimal_threshold(y_true, y_proba):
    precision, recall, thresholds = \
        precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / \
                (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    return thresholds[optimal_idx], f1_scores[optimal_idx]

models_info = {
    'Logistic Regression': lr_proba,
    'Random Forest':       rf_proba,
    'XGBoost':             xgb_proba
}

thresholds = {}
for name, proba in models_info.items():
    thresh, f1 = find_optimal_threshold(y_test, proba)
    thresholds[name] = thresh
    print(f"{name:<25} "
          f"Optimal threshold: {thresh:.3f}  "
          f"Best F1: {f1:.4f}")

# ===============================
# 3. PERFORMANCE AT OPTIMAL THRESHOLD
# ===============================
print("\n--- Performance at Optimal Threshold ---\n")

results = {}
for name, proba in models_info.items():
    thresh = thresholds[name]
    pred_opt = (proba >= thresh).astype(int)
    tp = ((pred_opt==1) & (y_test==1)).sum()
    fp = ((pred_opt==1) & (y_test==0)).sum()
    fn = ((pred_opt==0) & (y_test==1)).sum()
    tn = ((pred_opt==0) & (y_test==0)).sum()

    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1        = 2 * precision * recall / \
                (precision + recall) \
                if (precision+recall) > 0 else 0
    auc  = roc_auc_score(y_test, proba)
    ap   = average_precision_score(y_test, proba)

    results[name] = {
        'AUC':       auc,
        'AP':        ap,
        'Precision': precision,
        'Recall':    recall,
        'F1':        f1,
        'TP':        tp,
        'FP':        fp,
        'FN':        fn,
        'TN':        tn,
        'Threshold': thresh
    }

    print(f"{name}:")
    print(f"  ROC-AUC:         {auc:.4f}")
    print(f"  Avg Precision:   {ap:.4f}")
    print(f"  Fraud Precision: {precision:.4f}")
    print(f"  Fraud Recall:    {recall:.4f}")
    print(f"  Fraud F1-Score:  {f1:.4f}")
    print(f"  Frauds Caught:   {tp} / {tp+fn} "
          f"({recall*100:.1f}%)")
    print(f"  False Alarms:    {fp:,}")
    print()

# ===============================
# 4. BUSINESS IMPACT ANALYSIS
# ===============================
print("--- Business Impact Analysis ---")
avg_fraud_amount = 122.21
print(f"Assumed average fraud transaction: "
      f"${avg_fraud_amount:.2f}")
print(f"Total fraud cases in test set: "
      f"{y_test.sum()}")
print()

for name, r in results.items():
    caught_value = r['TP'] * avg_fraud_amount
    missed_value = r['FN'] * avg_fraud_amount
    print(f"{name}:")
    print(f"  Fraud value caught:  "
          f"${caught_value:,.2f}")
    print(f"  Fraud value missed:  "
          f"${missed_value:,.2f}")
    print(f"  False alerts raised: {r['FP']:,}")
    print()

# ===============================
# 5. VISUALIZATIONS
# ===============================
print("--- Generating Evaluation Plots ---")

# Plot 1: Threshold vs F1 Score
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#2196F3', '#4CAF50', '#F44336']
for ax, (name, proba), color in zip(
        axes, models_info.items(), colors):
    precision, recall, thresh = \
        precision_recall_curve(y_test, proba)
    f1 = 2*precision*recall / \
         (precision+recall+1e-8)
    ax.plot(thresh, f1[:-1], color=color,
            linewidth=2, label='F1 Score')
    ax.plot(thresh, precision[:-1],
            color='orange', linewidth=1.5,
            linestyle='--', label='Precision')
    ax.plot(thresh, recall[:-1],
            color='green', linewidth=1.5,
            linestyle=':', label='Recall')
    ax.axvline(x=thresholds[name],
               color='black', linestyle='--',
               linewidth=1.5,
               label=f"Optimal ({thresholds[name]:.3f})")
    ax.set_title(f'{name}\nThreshold Analysis',
                 fontweight='bold')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

plt.suptitle(
    'Threshold Analysis — Precision, Recall and F1 '
    'vs Decision Threshold\n'
    'Optimal threshold maximises F1-Score on '
    'fraudulent class',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/threshold_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/threshold_analysis.png")

# Plot 2: Business Impact
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

model_names  = list(results.keys())
short_names  = ['LR', 'RF', 'XGB']
caught_vals  = [r['TP']*avg_fraud_amount
                for r in results.values()]
missed_vals  = [r['FN']*avg_fraud_amount
                for r in results.values()]
false_alerts = [r['FP'] for r in results.values()]

x = np.arange(len(short_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, caught_vals,
                    width, label='Fraud Value Caught ($)',
                    color='#4CAF50', edgecolor='black',
                    linewidth=0.8)
bars2 = axes[0].bar(x + width/2, missed_vals,
                    width, label='Fraud Value Missed ($)',
                    color='#F44336', edgecolor='black',
                    linewidth=0.8)
for bar in bars1:
    axes[0].text(
        bar.get_x() + bar.get_width()/2.,
        bar.get_height() + 20,
        f'${bar.get_height():,.0f}',
        ha='center', fontsize=9, fontweight='bold'
    )
for bar in bars2:
    axes[0].text(
        bar.get_x() + bar.get_width()/2.,
        bar.get_height() + 20,
        f'${bar.get_height():,.0f}',
        ha='center', fontsize=9, fontweight='bold'
    )
axes[0].set_title(
    'Business Impact — Fraud Value Caught vs Missed\n'
    f'(Avg fraud transaction: ${avg_fraud_amount})',
    fontweight='bold'
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names)
axes[0].set_ylabel('Dollar Value ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(short_names, false_alerts,
            color='#FF9800', edgecolor='black',
            linewidth=0.8, width=0.4)
for i, val in enumerate(false_alerts):
    axes[1].text(i, val + 50, f'{val:,}',
                 ha='center', fontweight='bold',
                 fontsize=11)
axes[1].set_title(
    'False Positive Count\n'
    '(Legitimate transactions incorrectly '
    'flagged as fraud)',
    fontweight='bold'
)
axes[1].set_ylabel('Number of False Alerts')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle(
    'Business Impact Analysis — '
    'Financial Value of Model Performance',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/business_impact.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/business_impact.png")

# Plot 3: Complete Dashboard
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.4, wspace=0.35)

# ROC Curves
ax1 = fig.add_subplot(gs[0, :2])
for (name, proba), color in zip(
        models_info.items(),
        ['#2196F3', '#4CAF50', '#F44336']):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax1.plot(fpr, tpr, linewidth=2.5,
             label=f'{name} (AUC={auc:.4f})',
             color=color)
ax1.plot([0,1],[0,1],'k--',linewidth=1)
ax1.set_title('ROC Curves — All Models',
              fontweight='bold', fontsize=12)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Model Metrics
ax2 = fig.add_subplot(gs[0, 2])
metrics = ['AUC', 'AP', 'Recall', 'F1']
x = np.arange(len(metrics))
width = 0.25
colors_bar = ['#2196F3', '#4CAF50', '#F44336']
for i, (name, r) in enumerate(results.items()):
    vals = [r['AUC'], r['AP'],
            r['Recall'], r['F1']]
    ax2.bar(x + i*width, vals, width,
            label=name.split()[0],
            color=colors_bar[i],
            edgecolor='black', linewidth=0.5)
ax2.set_title('Model Metrics Comparison',
              fontweight='bold', fontsize=12)
ax2.set_xticks(x + width)
ax2.set_xticklabels(metrics)
ax2.set_ylabel('Score')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.1])

# PR Curves
ax3 = fig.add_subplot(gs[1, :2])
for (name, proba), color in zip(
        models_info.items(),
        ['#2196F3', '#4CAF50', '#F44336']):
    precision, recall, _ = \
        precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax3.plot(recall, precision, linewidth=2.5,
             label=f'{name} (AP={ap:.4f})',
             color=color)
ax3.axhline(y=y_test.mean(), color='gray',
            linestyle='--', linewidth=1.5,
            label='Baseline')
ax3.set_title(
    'Precision-Recall Curves — All Models',
    fontweight='bold', fontsize=12
)
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Score Distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(rf_proba[y_test==0], bins=40,
         alpha=0.7, color='#2196F3',
         label='Legitimate', density=True)
ax4.hist(rf_proba[y_test==1], bins=40,
         alpha=0.7, color='#F44336',
         label='Fraudulent', density=True)
ax4.axvline(x=thresholds['Random Forest'],
            color='black', linestyle='--',
            linewidth=2,
            label=f"Threshold "
                  f"({thresholds['Random Forest']:.3f})")
ax4.set_title('Score Distribution (RF)',
              fontweight='bold', fontsize=12)
ax4.set_xlabel('Fraud Probability')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle(
    'Credit Card Fraud Detection — '
    'Complete Evaluation Dashboard\n'
    'Thabiso Mdaka | BSc Electronic Engineering | UKZN',
    fontsize=14, fontweight='bold'
)
plt.savefig('outputs/plots/evaluation_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/evaluation_dashboard.png")

# ===============================
# 6. SAVE FULL REPORT
# ===============================
full_report = pd.DataFrame([
    {
        'Model': name,
        'ROC_AUC':   round(r['AUC'], 4),
        'Avg_Precision': round(r['AP'], 4),
        'Fraud_Precision': round(r['Precision'], 4),
        'Fraud_Recall':    round(r['Recall'], 4),
        'Fraud_F1':        round(r['F1'], 4),
        'Frauds_Caught':   r['TP'],
        'Frauds_Missed':   r['FN'],
        'False_Alerts':    r['FP'],
        'Optimal_Threshold': round(r['Threshold'], 4)
    }
    for name, r in results.items()
])
full_report.to_csv(
    'outputs/reports/full_evaluation.csv',
    index=False
)
print("Saved: outputs/reports/full_evaluation.csv")

print("\n" + "="*60)
print("   EVALUATION COMPLETE")
print("="*60)
print("\nPlots saved:")
print("  outputs/plots/threshold_analysis.png")
print("  outputs/plots/business_impact.png")
print("  outputs/plots/evaluation_dashboard.png")
print("="*60)