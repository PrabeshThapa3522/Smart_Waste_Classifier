import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]

IMAGE_SIZE = (224, 224)


class WasteClassifier:

    def __init__(self, model_path):
        """
        Load trained model
        """
        self.model = load_model(model_path)

    def preprocess_image(self, image_path):
        """
        Load and preprocess image
        """

        img = image.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        img_array = image.img_to_array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        return img_array

    def predict(self, image_path):

        img = self.preprocess_image(image_path)

        prediction = self.model.predict(img, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = np.max(prediction)

        return (
            CLASS_NAMES[predicted_index],
            float(confidence)
        )


if __name__ == "__main__":

    MODEL_PATH = "saved_model/mobilenet.keras"

    IMAGE_PATH = "sample.jpg"

    classifier = WasteClassifier(MODEL_PATH)

    label, confidence = classifier.predict(IMAGE_PATH)

    print("Prediction :", label)

    print(f"Confidence : {confidence*100:.2f}%")