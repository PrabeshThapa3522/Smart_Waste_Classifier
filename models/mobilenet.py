import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model


def build_mobilenet(input_shape=(224, 224, 3), num_classes=5):
    """
    MobileNetV2 Transfer Learning Model
    """

    # Load pretrained MobileNetV2 (trained on ImageNet)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    predictions = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions,
        name="MobileNetV2_WasteClassifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def fine_tune_model(model):
    """
    Unfreeze last layers for fine-tuning
    """

    base_model = model.layers[0]

    base_model.trainable = True

    # Freeze all except last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_mobilenet()
    model.summary()