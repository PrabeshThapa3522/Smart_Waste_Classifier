import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=5):
    """
    Custom CNN model for Smart Waste Classification

    Classes:
    0 - Glass
    1 - Metal
    2 - Organic
    3 - Paper
    4 - Plastic
    """

    model = Sequential(name="SmartWasteCNN")

    # -----------------------------
    # Block 1
    # -----------------------------
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Block 2
    # -----------------------------
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Block 3
    # -----------------------------
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Block 4
    # -----------------------------
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Classifier
    # -----------------------------
    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax"))

    # -----------------------------
    # Compile
    # -----------------------------
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    cnn = build_custom_cnn()
    cnn.summary()