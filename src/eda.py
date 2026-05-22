import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs/plots', exist_ok=True)

print("=" * 60)
print("   FRAUD DETECTION — Exploratory Data Analysis")
print("=" * 60)

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv('data/raw/creditcard.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Transactions:  {len(df):,}")
print(f"Fraudulent:          {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
print(f"Legitimate:          {(df['Class']==0).sum():,} ({(1-df['Class'].mean())*100:.3f}%)")
print(f"\nMissing Values:\n{df.isnull().sum().sum()} total missing values")
print(f"\nFeature Range:")
print(f"  Amount: {df['Amount'].min():.2f} to {df['Amount'].max():.2f}")
print(f"  Time:   {df['Time'].min():.0f} to {df['Time'].max():.0f} seconds")

# ===============================
# 2. CLASS DISTRIBUTION
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

counts = df['Class'].value_counts()
bars = axes[0].bar(
    ['Legitimate', 'Fraudulent'],
    counts.values,
    color=['#2196F3', '#F44336'],
    edgecolor='black', linewidth=0.8, width=0.5
)
for bar, count in zip(bars, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width()/2.,
        bar.get_height() + 1000,
        f'{count:,}', ha='center',
        fontweight='bold', fontsize=12
    )
axes[0].set_title(
    'Class Distribution\nSevere Class Imbalance',
    fontweight='bold', fontsize=12
)
axes[0].set_ylabel('Number of Transactions')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, counts.max() * 1.15)

axes[1].pie(
    counts.values,
    labels=[f'Legitimate\n({(1-df["Class"].mean())*100:.2f}%)',
            f'Fraudulent\n({df["Class"].mean()*100:.2f}%)'],
    colors=['#2196F3', '#F44336'],
    explode=(0, 0.1),
    autopct='%1.3f%%',
    startangle=90,
    textprops={'fontsize': 11}
)
axes[1].set_title(
    'Proportion of Fraud\nvs Legitimate Transactions',
    fontweight='bold', fontsize=12
)

plt.suptitle(
    'Credit Card Fraud Dataset — Class Imbalance Analysis\n'
    '284,807 Transactions | 492 Fraudulent (0.173%)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/class_distribution.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("\nSaved: outputs/plots/class_distribution.png")

# ===============================
# 3. TRANSACTION AMOUNT ANALYSIS
# ===============================
fraud = df[df['Class'] == 1]['Amount']
legit = df[df['Class'] == 0]['Amount']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram
axes[0].hist(legit.clip(upper=500), bins=50,
             color='#2196F3', alpha=0.7,
             label='Legitimate', density=True)
axes[0].hist(fraud.clip(upper=500), bins=50,
             color='#F44336', alpha=0.7,
             label='Fraudulent', density=True)
axes[0].set_title('Transaction Amount Distribution\n(Clipped at $500)',
                  fontweight='bold')
axes[0].set_xlabel('Transaction Amount ($)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Boxplot
bp = axes[1].boxplot(
    [legit.clip(upper=500), fraud.clip(upper=500)],
    labels=['Legitimate', 'Fraudulent'],
    patch_artist=True,
    boxprops=dict(linewidth=1.5),
    medianprops=dict(color='black', linewidth=2)
)
bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#F44336')
bp['boxes'][1].set_alpha(0.7)
axes[1].set_title('Amount Boxplot\n(Clipped at $500)',
                  fontweight='bold')
axes[1].set_ylabel('Transaction Amount ($)')
axes[1].grid(True, alpha=0.3)

# Summary statistics table
stats_data = {
    'Metric': ['Mean', 'Median', 'Std Dev',
               'Min', 'Max', 'Count'],
    'Legitimate': [
        f'${legit.mean():.2f}',
        f'${legit.median():.2f}',
        f'${legit.std():.2f}',
        f'${legit.min():.2f}',
        f'${legit.max():.2f}',
        f'{len(legit):,}'
    ],
    'Fraudulent': [
        f'${fraud.mean():.2f}',
        f'${fraud.median():.2f}',
        f'${fraud.std():.2f}',
        f'${fraud.min():.2f}',
        f'${fraud.max():.2f}',
        f'{len(fraud):,}'
    ]
}
axes[2].axis('off')
table = axes[2].table(
    cellText=list(zip(
        stats_data['Metric'],
        stats_data['Legitimate'],
        stats_data['Fraudulent']
    )),
    colLabels=['Metric', 'Legitimate', 'Fraudulent'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1F4E79')
        cell.set_text_props(color='white',
                            fontweight='bold')
    elif col == 2:
        cell.set_facecolor('#FFEBEE')
    elif col == 1:
        cell.set_facecolor('#E3F2FD')
axes[2].set_title('Descriptive Statistics',
                  fontweight='bold', pad=20)

plt.suptitle(
    'Transaction Amount Analysis — Fraud vs Legitimate',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/amount_analysis.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/amount_analysis.png")

# ===============================
# 4. TIME ANALYSIS
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df['Hour'] = (df['Time'] / 3600) % 24

fraud_hours = df[df['Class']==1]['Hour']
legit_hours = df[df['Class']==0]['Hour']

axes[0].hist(legit_hours, bins=24, color='#2196F3',
             alpha=0.7, label='Legitimate',
             density=True)
axes[0].hist(fraud_hours, bins=24, color='#F44336',
             alpha=0.7, label='Fraudulent',
             density=True)
axes[0].set_title('Transaction Volume by Hour of Day',
                  fontweight='bold')
axes[0].set_xlabel('Hour of Day (0-24)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

hourly_fraud_rate = df.groupby(
    df['Hour'].astype(int)
)['Class'].mean() * 100
axes[1].bar(hourly_fraud_rate.index,
            hourly_fraud_rate.values,
            color='#F44336', alpha=0.7,
            edgecolor='black', linewidth=0.5)
axes[1].set_title('Fraud Rate by Hour of Day (%)',
                  fontweight='bold')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle(
    'Temporal Analysis — When Does Fraud Occur?',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/time_analysis.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/time_analysis.png")

# ===============================
# 5. PCA FEATURE ANALYSIS
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

top_features = ['V14', 'V17', 'V12', 'V10']
for ax, feat in zip(axes.flatten(), top_features):
    fraud_vals = df[df['Class']==1][feat]
    legit_vals = df[df['Class']==0][feat]
    ax.hist(legit_vals, bins=50, color='#2196F3',
            alpha=0.7, label='Legitimate',
            density=True)
    ax.hist(fraud_vals, bins=50, color='#F44336',
            alpha=0.7, label='Fraudulent',
            density=True)
    ax.set_title(f'Feature {feat} Distribution',
                 fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Top PCA Feature Distributions — Fraud vs Legitimate\n'
    'Features V14, V17, V12, V10 show strongest separation',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/feature_distributions.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/feature_distributions.png")

# ===============================
# 6. CORRELATION HEATMAP
# ===============================
fraud_df = df[df['Class']==1].drop(
    columns=['Time', 'Class']
)
corr = fraud_df.corr()

plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, cmap='RdBu_r',
    center=0, vmin=-1, vmax=1,
    linewidths=0.3, annot=False,
    cbar_kws={'shrink': 0.8}
)
plt.title(
    'Feature Correlation Matrix — Fraudulent Transactions Only\n'
    'PCA features V1-V28 are by design uncorrelated '
    '(orthogonal)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('outputs/plots/correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/plots/correlation_heatmap.png")

print("\n" + "="*60)
print("   EDA COMPLETE — 5 plots saved to outputs/plots/")
print("="*60)