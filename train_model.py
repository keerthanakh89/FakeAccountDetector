import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_DIR / "train.csv")

print("✅ Dataset loaded successfully!")
print("Columns:", list(df.columns))
print("Number of rows:", len(df))

# =========================
# Select relevant features
# =========================
# These are the main numeric and useful columns from your dataset
X = df[['#followers', '#follows', '#posts', 'private']].fillna(0)

# =========================
# Target column (label)
# =========================
# 'fake' column contains 1 for fake and 0 for real
y = df['fake'].astype(int)

# =========================
# Split data into train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Initialize Models
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42, class_weight="balanced")
}

# =========================
# Train and Evaluate
# =========================
results = {}
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    results[name] = acc

# =========================
# Find Best Model
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, MODEL_DIR / "fake_model.pkl")
print("💾 Model saved successfully at: model/fake_model.pkl")

# =========================
# Plot Comparison Graph
# =========================
plt.figure(figsize=(6, 4))
plt.bar(results.keys(), results.values(), color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(MODEL_DIR / "model_comparison.png")
plt.close()
print("📊 Accuracy comparison chart saved as: model/model_comparison.png")

print("\n🎉 Training complete! You can now run: python app.py")
































