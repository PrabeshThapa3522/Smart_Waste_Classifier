import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from utils.preprocessing import load_data
from models.custom_cnn import build_custom_cnn
from models.mobilenet import build_mobilenet

# ==============================
# Configuration
# ==============================

DATASET_PATH = "dataset"

EPOCHS = 15

MODEL_DIR = "saved_model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# Load Dataset
# ==============================

train_data, validation_data, test_data = load_data(DATASET_PATH)

NUM_CLASSES = train_data.num_classes

print("\nClass Labels")
print(train_data.class_indices)

# ==============================
# Callbacks
# ==============================

checkpoint_cnn = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_DIR,
        "custom_cnn.keras"
    ),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

checkpoint_mobile = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_DIR,
        "mobilenet.keras"
    ),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

# ==============================
# Build Models
# ==============================

print("\nBuilding Custom CNN...")
cnn = build_custom_cnn(num_classes=NUM_CLASSES)

print("\nBuilding MobileNetV2...")
mobilenet = build_mobilenet(num_classes=NUM_CLASSES)

# ==============================
# Train Custom CNN
# ==============================

print("\nTraining Custom CNN...\n")

history_cnn = cnn.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_cnn,
        early_stop,
        reduce_lr
    ]
)

# ==============================
# Train MobileNetV2
# ==============================

print("\nTraining MobileNetV2...\n")

history_mobile = mobilenet.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[
        checkpoint_mobile,
        early_stop,
        reduce_lr
    ]
)

# ==============================
# Plot Function
# ==============================

def plot_history(history, title):

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.title(title + " Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    # Loss
    plt.subplot(1,2,2)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.title(title + " Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(["Train","Validation"])

    plt.tight_layout()

    plt.show()

# ==============================
# Plot Results
# ==============================

plot_history(history_cnn, "Custom CNN")

plot_history(history_mobile, "MobileNetV2")

print("\nTraining Completed Successfully!")

print("\nSaved Models:")

print("saved_model/custom_cnn.keras")

print("saved_model/mobilenet.keras")