# preprocessing.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def load_data(dataset_path):

    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    validation_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )

    test_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_data = train_generator.flow_from_directory(
        dataset_path + "/train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    validation_data = validation_generator.flow_from_directory(
        dataset_path + "/validation",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_generator.flow_from_directory(
        dataset_path + "/test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, validation_data, test_data