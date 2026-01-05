import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diabetes_model import DiabetesModel
from src.data_loader import load_and_prepare_data


# =============================
# LOAD & PREPARE DATA
# =============================
print("Loading data WITH interaction terms...")
X_train, X_test, y_train, y_test = load_and_prepare_data(add_interactions=True)

# =============================
# TRAIN MODEL
# =============================
model = DiabetesModel()
model.train(X_train, y_train)

# =============================
# BASELINE EVALUATION (DEFAULT THRESHOLD = 0.5)
# =============================
print("\n Baseline evaluation (threshold = 0.5)")
model.evaluate(X_test, y_test)

# =============================
# PROBABILITIES
# =============================
y_proba = model.model.predict_proba(X_test)[:, 1]

# =============================
# PRECISION-RECALL CURVE
# =============================
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.grid()
plt.savefig('precision_recall_curve.png')
print("\n📈 Precision-Recall curve saved to 'precision_recall_curve.png'")
plt.close()
print("\n To shown that blind thresholding is suboptimal. ")

# =============================
# FIND "OPTIMAL" THRESHOLD (MAX F1)
# =============================
f1_scores = 2 * (precision * recall) / (precision + recall)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n Optimal threshold (max F1-score): {best_threshold:.2f}")

# =============================
# EVALUATE WITH OPTIMAL THRESHOLD
# =============================
y_pred_optimal = (y_proba >= best_threshold).astype(int)

print(f"\n🧪 Evaluation with threshold = {best_threshold:.2f}")
print(classification_report(y_test, y_pred_optimal))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba)}")
print(f"\n This shows that by tuning the threshold based on F1-score, we can achieve a better balance (theoretically) between precision and recall compared to the default 0.5 threshold. But its clinically false. F1 got tricked by imbalance.")

# =============================
# CLINICAL THRESHOLD (RECALL-BASED). 
# Best one for medical use
# =============================
print("\n" + "="*60)
print("CLINICAL APPROACH: Setting Minimum Recall Target")
print("="*60)

# Step 1: Remove NaNs from precision/recall
valid = ~np.isnan(precision) & ~np.isnan(recall)
precision_clean = precision[valid]
recall_clean = recall[valid]
thresholds_clean = thresholds[:len(precision_clean)]

# Step 2: Choose medical target (i want it to catch at least 85% of diabetics, but medical experts may set this differently)
TARGET_RECALL = 0.85

# Step 3: Find the threshold that achieves it
valid_indices = np.where(recall_clean >= TARGET_RECALL)[0]
if len(valid_indices) > 0:
    clinical_idx = valid_indices[-1]  # Get the LAST one (highest threshold)
    clinical_threshold = thresholds_clean[clinical_idx]
else:
    clinical_idx = 0  # Fallback
    clinical_threshold = thresholds_clean[0]

print(f"\n🎯 Clinically chosen threshold (recall ≥ {TARGET_RECALL}): {clinical_threshold:.2f}")
print(f"   At this threshold:")
print(f"   - Recall: {recall_clean[clinical_idx]:.2%}")
print(f"   - Precision: {precision_clean[clinical_idx]:.2%}")

# Step 4: Evaluate with clinical threshold
y_pred_clinical = (y_proba >= clinical_threshold).astype(int)

print(f"\n🩺 Clinical evaluation (threshold = {clinical_threshold:.2f})")
print(classification_report(y_test, y_pred_clinical))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# =============================
# FEATURE IMPORTANCE
# =============================
print("\n" + "="*60)
print("🔍 Top Contributing Features")
print("="*60)
top_features = model.feature_importance(X_train.columns, top_k=10)

for i, (feature, weight) in enumerate(top_features, 1):
    direction = "↑ Increases" if weight > 0 else "↓ Decreases"
    print(f"{i:2d}. {feature:20s}: {weight:+.3f}  {direction} diabetes risk")
