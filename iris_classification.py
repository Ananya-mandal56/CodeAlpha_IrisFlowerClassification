"""
╔══════════════════════════════════════════════════════════════╗
║         CODEALPHA INTERNSHIP — DATA SCIENCE                  ║
║         TASK 1: Iris Flower Classification                   ║
╚══════════════════════════════════════════════════════════════╝

Dataset  : Iris (built-in via sklearn)
Models   : Logistic Regression, KNN, Decision Tree, Random Forest, SVM
Goal     : Classify Iris species (setosa / versicolor / virginica)
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ── 2. Load & Explore Dataset ─────────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 60)
print("  IRIS DATASET — OVERVIEW")
print("=" * 60)
print(f"\n  Shape        : {df.shape}")
print(f"  Features     : {list(iris.feature_names)}")
print(f"  Target       : {list(iris.target_names)}")
print(f"  Class counts :\n{df['species'].value_counts().to_string()}")
print(f"\n  Missing values: {df.isnull().sum().sum()}")
print("\n  First 5 rows:")
print(df.head().to_string(index=False))
print("\n  Statistical Summary:")
print(df.describe().round(2).to_string())

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 4. Define Models (all wrapped in pipelines with StandardScaler) ───────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

pipelines = {
    name: Pipeline([("scaler", StandardScaler()), ("model", clf)])
    for name, clf in models.items()
}

# ── 5. Cross-Validation + Test Evaluation ────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
print("\n" + "=" * 60)
print("  MODEL EVALUATION")
print("=" * 60)

for name, pipe in pipelines.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std" : cv_scores.std(),
        "test_acc": test_acc,
        "y_pred" : y_pred,
    }
    print(f"\n  {name}")
    print(f"    CV Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Test Accuracy: {test_acc:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["test_acc"])
best_pipe  = pipelines[best_name]
best_pred  = results[best_name]["y_pred"]

print(f"\n{'='*60}")
print(f"  ★  Best Model : {best_name}")
print(f"     Test Acc   : {results[best_name]['test_acc']:.4f}")
print(f"{'='*60}")
print(f"\n  Classification Report ({best_name}):\n")
print(classification_report(y_test, best_pred, target_names=iris.target_names))

# ── 6. Visualisations ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
COLORS = sns.color_palette("Set2", 3)

# ── Fig 1: EDA (pairplot + class distribution + correlation) ──────────────────
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Iris Dataset — Exploratory Data Analysis", fontsize=15, fontweight="bold")

# Class distribution
species_counts = df["species"].value_counts()
axes[0].bar(species_counts.index, species_counts.values, color=COLORS, edgecolor="white", linewidth=1.5)
axes[0].set_title("Class Distribution")
axes[0].set_xlabel("Species")
axes[0].set_ylabel("Count")
for i, v in enumerate(species_counts.values):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

# Feature correlation heatmap
corr = df.drop("species", axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1],
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
axes[1].set_title("Feature Correlation Heatmap")
plt.tight_layout()
fig1.savefig("/home/claude/fig1_eda.png", dpi=150, bbox_inches="tight")
print("\n  Saved: fig1_eda.png")

# ── Fig 2: Feature distributions by species ───────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle("Feature Distributions by Species", fontsize=15, fontweight="bold")

for ax, feature in zip(axes2.flat, iris.feature_names):
    for species, color in zip(iris.target_names, COLORS):
        subset = df[df["species"] == species][feature]
        ax.hist(subset, bins=15, alpha=0.6, label=species, color=color, edgecolor="white")
    ax.set_title(feature.replace(" (cm)", "").title())
    ax.set_xlabel("cm")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

plt.tight_layout()
fig2.savefig("/home/claude/fig2_feature_dist.png", dpi=150, bbox_inches="tight")
print("  Saved: fig2_feature_dist.png")

# ── Fig 3: Model comparison ───────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Model Comparison", fontsize=15, fontweight="bold")

names     = list(results.keys())
cv_means  = [results[n]["cv_mean"]  for n in names]
cv_stds   = [results[n]["cv_std"]   for n in names]
test_accs = [results[n]["test_acc"] for n in names]
short_names = ["LR", "KNN", "DT", "RF", "SVM"]

bar_colors = ["#2ecc71" if n == best_name else "#95a5a6" for n in names]

# CV accuracy
axes3[0].barh(short_names, cv_means, xerr=cv_stds, color=bar_colors,
              edgecolor="white", linewidth=1.2, capsize=4)
axes3[0].set_xlim(0.85, 1.01)
axes3[0].set_title("5-Fold CV Accuracy")
axes3[0].set_xlabel("Accuracy")
for i, (v, e) in enumerate(zip(cv_means, cv_stds)):
    axes3[0].text(v + e + 0.001, i, f"{v:.3f}", va="center", fontsize=9)

# Test accuracy
axes3[1].barh(short_names, test_accs, color=bar_colors, edgecolor="white", linewidth=1.2)
axes3[1].set_xlim(0.85, 1.01)
axes3[1].set_title("Test Accuracy")
axes3[1].set_xlabel("Accuracy")
for i, v in enumerate(test_accs):
    axes3[1].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)

plt.tight_layout()
fig3.savefig("/home/claude/fig3_model_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: fig3_model_comparison.png")

# ── Fig 4: Confusion matrix of best model ────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax4, colorbar=False, cmap="Blues")
ax4.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
plt.tight_layout()
fig4.savefig("/home/claude/fig4_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("  Saved: fig4_confusion_matrix.png")

# ── Fig 5: Feature importances (Random Forest) ───────────────────────────────
rf_pipe = pipelines["Random Forest"]
importances = rf_pipe.named_steps["model"].feature_importances_
feat_df = pd.DataFrame({"Feature": iris.feature_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

fig5, ax5 = plt.subplots(figsize=(7, 4))
ax5.barh(feat_df["Feature"], feat_df["Importance"],
         color=sns.color_palette("viridis", len(feat_df)), edgecolor="white")
ax5.set_title("Random Forest — Feature Importances", fontweight="bold")
ax5.set_xlabel("Importance Score")
for i, v in enumerate(feat_df["Importance"]):
    ax5.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
fig5.savefig("/home/claude/fig5_feature_importance.png", dpi=150, bbox_inches="tight")
print("  Saved: fig5_feature_importance.png")

# ── 7. Sample Predictions ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS (first 10 test samples)")
print("=" * 60)
sample_df = pd.DataFrame(X_test[:10], columns=iris.feature_names)
sample_df["Actual"]    = [iris.target_names[i] for i in y_test[:10]]
sample_df["Predicted"] = [iris.target_names[i] for i in best_pred[:10]]
sample_df["Correct"]   = sample_df["Actual"] == sample_df["Predicted"]
print(sample_df.to_string(index=False))

print("\n  ✅ All visualisations saved successfully.")
print("  ✅ Script complete.\n")
