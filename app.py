import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ==========================
# Configuration
# ==========================

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

MODEL_PATH = "saved_model/mobilenet.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]

# ==========================
# Load Model
# ==========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================
# Title
# ==========================

st.title("♻️ Smart Waste Classifier")

st.markdown(
    """
Upload an image of waste and the model will classify it into one of the following categories:

- 🟢 Cardboard
- 🍾 Glass
- 🔩 Metal
- 📄 Paper
- 🥤 Plastic
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# Prediction
# ==========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    img = image.resize(IMAGE_SIZE)

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    confidence = np.max(prediction)

    st.success(
        f"Prediction: {predicted_class.upper()}"
    )

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):

        st.progress(float(prediction[0][i]))

        st.write(
            f"{class_name.capitalize()} : {prediction[0][i]*100:.2f}%"
        )