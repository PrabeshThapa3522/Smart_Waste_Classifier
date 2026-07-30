import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils.preprocessing import load_data

# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "dataset"

# ==========================================
# Load Test Data
# ==========================================

_, _, test_data = load_data(DATASET_PATH)

# ==========================================
# Parse Command Line Argument
# ==========================================

parser = argparse.ArgumentParser(
    description="Evaluate Waste Classification Model"
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to saved model (.keras)"
)

args = parser.parse_args()

# ==========================================
# Load Model
# ==========================================

print(f"\nLoading model: {args.model}")

model = load_model(args.model)

# ==========================================
# Prediction
# ==========================================

print("\nPredicting...")

predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)

y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

# ==========================================
# Metrics
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\n===============================")
print("Evaluation Results")
print("===============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.show()