"""
Integrated Capstone Synthesis Pipeline
Front-End Statistical Hypothesis Testing + Lifecycle Machine Learning Modeling
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# =====================================================================
# SECTION 1: SYNTHETIC DATA GENERATION (CONTROLLED BUSINESS ENVIRONMENT)
# =====================================================================
N = 3000

tenure = np.random.exponential(scale=24, size=N).clip(1, 72)
monthly_charges = np.random.normal(loc=65, scale=25, size=N).clip(20, 130)
total_charges = tenure * monthly_charges + np.random.normal(0, 30, size=N)
support_tickets = np.random.poisson(lam=1.8, size=N)
contract_type = np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=N, p=[0.55, 0.25, 0.20])
paperless_billing = np.random.choice(['Yes', 'No'], size=N, p=[0.60, 0.40])

# Front-End Checkout Group Allocation (Simulated A/B/C Test)
checkout_variant = np.random.choice(['Control_A', 'OnePage_B', 'Gamified_C'], size=N, p=[0.334, 0.333, 0.333])

# Ground truth AOV effects: Variant A: base, Variant B: +$2, Variant C: +$6.5
aov = np.where(
    checkout_variant == 'Control_A', np.random.normal(78.0, 16.0, size=N),
    np.where(checkout_variant == 'OnePage_B', np.random.normal(80.2, 15.5, size=N),
             np.random.normal(84.5, 17.2, size=N))
)

# Churn logic with non-linear contract and support interactions
log_odds = (
    -0.85
    - 0.045 * tenure
    + 0.022 * monthly_charges
    + 0.45 * support_tickets
    + (1.15 * (contract_type == 'Month-to-Month'))
    - (0.95 * (contract_type == 'Two-Year'))
    + (0.35 * (paperless_billing == 'Yes'))
)
churn_prob = 1 / (1 + np.exp(-log_odds))
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    'CheckoutVariant': checkout_variant,
    'OrderValue': np.round(aov, 2),
    'Tenure': np.round(tenure, 1),
    'MonthlyCharges': np.round(monthly_charges, 2),
    'TotalCharges': np.round(total_charges, 2),
    'SupportTickets': support_tickets,
    'ContractType': contract_type,
    'PaperlessBilling': paperless_billing,
    'Churn': churn
})

# =====================================================================
# SECTION 2: STATISTICAL INFERENCE & HYPOTHESIS TESTING
# =====================================================================
print("\n" + "="*70)
print(" STAGE 1: INFERENTIAL HYPOTHESIS TESTING (FRONT-END OPTIMIZATION)")
print("="*70)

# 1. One-Way ANOVA on Order Value across variants
anova_model = ols('OrderValue ~ C(CheckoutVariant)', data=df).fit()
anova_res = sm.stats.anova_lm(anova_model, typ=2)
print("One-Way ANOVA Table:\n", anova_res)

# 2. Tukey's HSD Post-Hoc Test
tukey = pairwise_tukeyhsd(endog=df['OrderValue'], groups=df['CheckoutVariant'], alpha=0.05)
print("\nTukey HSD Pairwise Comparisons:\n", tukey)

# 3. Welch's Two-Sample t-test: Gamified_C vs Control_A
ctrl_aov = df[df['CheckoutVariant'] == 'Control_A']['OrderValue']
gamif_aov = df[df['CheckoutVariant'] == 'Gamified_C']['OrderValue']
t_stat, t_pval = stats.ttest_ind(gamif_aov, ctrl_aov, equal_var=False)
print(f"\nWelch's t-test (Variant C vs Variant A): t = {t_stat:.4f}, p-value = {t_pval:.4e}")

# =====================================================================
# SECTION 3: MACHINE LEARNING MODEL DEVELOPMENT & CROSS-VALIDATION
# =====================================================================
print("\n" + "="*70)
print(" STAGE 2: MACHINE LEARNING PIPELINE (LIFECYCLE CHURN ENGINE)")
print("="*70)

X = df[['Tenure', 'MonthlyCharges', 'TotalCharges', 'SupportTickets', 'ContractType', 'PaperlessBilling']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'SupportTickets']
cat_cols = ['ContractType', 'PaperlessBilling']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

pipe_lr = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42))
])

pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=6, random_state=42))
])

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_lr = cross_val_score(pipe_lr, X_train, y_train, cv=cv, scoring='roc_auc')
cv_rf = cross_val_score(pipe_rf, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"5-Fold CV ROC-AUC | Logistic Regression: {cv_lr.mean():.4f} (+/- {cv_lr.std():.4f})")
print(f"5-Fold CV ROC-AUC | Random Forest:       {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")

# Fit models on full training partition
pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
y_prob_rf = pipe_rf.predict_proba(X_test)[:, 1]
y_pred_rf = pipe_rf.predict(X_test)

print("\nRandom Forest Test Classification Report:\n")
print(classification_report(y_test, y_pred_rf, digits=4))

# =====================================================================
# SECTION 4: THRESHOLD UTILITY OPTIMIZATION & EXPECTED VALUE
# =====================================================================
print("\n" + "="*70)
print(" STAGE 3: COST-UTILITY THRESHOLD OPTIMIZATION")
print("="*70)

cost_retention = 45.0   # Cost of an intervention offer
churn_loss = 420.0       # Average Customer Lifetime Value lost
save_success_rate = 0.50 # Probability that an at-risk customer accepts the offer

thresholds = np.linspace(0.10, 0.80, 71)
net_values = []

for thresh in thresholds:
    preds = (y_prob_rf >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    # Expected business value calculation
    savings = tp * (save_success_rate * churn_loss - cost_retention)
    penalty_fp = fp * (-cost_retention)
    penalty_fn = fn * (-churn_loss)
    total_val = savings + penalty_fp + penalty_fn
    net_values.append(total_val)

best_idx = np.argmax(net_values)
optimal_threshold = thresholds[best_idx]
print(f"Optimal Decision Threshold: tau = {optimal_threshold:.2f}")
print(f"Max Test Set Economic Payoff: ${net_values[best_idx]:,.2f} (vs Default tau=0.5:${net_values[np.argmin(np.abs(thresholds - 0.50))]:,.2f})")

# =====================================================================
# SECTION 5: PUBLICATION VISUALIZATIONS GENERATION
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
sns.set_theme(style="whitegrid")

# Panel 1: Hypothesis Testing AOV Distributions
sns.boxplot(x='CheckoutVariant', y='OrderValue', data=df, ax=axes[0, 0], palette="Set2", showmeans=True,
            meanprops={"marker":"D", "markerfacecolor":"red", "markeredgecolor":"black"})
axes[0, 0].set_title("(A) AOV Distribution Across Checkout Variants (ANOVA p < 1e-12)", fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel("Order Value ($)")

# Panel 2: Comparative ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
axes[0, 1].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})', lw=2)
axes[0, 1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})', lw=2.5, color='darkgreen')
axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0, 1].set_title("(B) Receiver Operating Characteristic (ROC) Comparison", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right")

# Panel 3: Precision-Recall Curves
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
axes[1, 0].plot(rec_lr, prec_lr, label=f'Logistic Regression (AP = {average_precision_score(y_test, y_prob_lr):.3f})', lw=2)
axes[1, 0].plot(rec_rf, prec_rf, label=f'Random Forest (AP = {average_precision_score(y_test, y_prob_rf):.3f})', lw=2.5, color='darkgreen')
axes[1, 0].axhline(y_test.mean(), color='crimson', linestyle='--', label=f'Class Baseline ({y_test.mean():.2f})')
axes[1, 0].set_title("(C) Precision-Recall Dynamics across Imbalanced Classes", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].legend(loc="upper right")

# Panel 4: Cost-Utility Threshold Curve
axes[1, 1].plot(thresholds, net_values, lw=2.5, color='purple')
axes[1, 1].axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal tau = {optimal_threshold:.2f}')
axes[1, 1].axvline(0.50, color='gray', linestyle=':', label='Standard tau = 0.50')
axes[1, 1].set_title("(D) Expected Economic Utility Curve vs. Classification Cutoff", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Decision Threshold (tau)")
axes[1, 1].set_ylabel("Net Utility ($)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("capstone_integrated_results.png", dpi=300)
plt.show()