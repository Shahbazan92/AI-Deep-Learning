import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Skin Cancer Image Classifier",
    page_icon="🔬",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "skin_cancer_detector.keras"
IMG_SIZE = (128, 128)

# IMPORTANT:
# These class names must match the alphabetical folder order used by
# image_dataset_from_directory() during training.
# If your training folders were named differently, update this list.
CLASS_NAMES = ["benign", "malignant"]

@st.cache_resource
def load_cnn_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload skin_cancer_detector.keras to the same GitHub repository "
            "as this app.py file."
        )
    return tf.keras.models.load_model(MODEL_PATH)

st.title("Skin Cancer Image Classifier")
st.write(
    "Upload a skin-lesion image to get a prediction from your trained CNN model."
)

st.warning(
    "Educational use only. This model is not a medical diagnosis and must not "
    "be used to decide whether a lesion is cancerous. Consult a qualified "
    "healthcare professional for medical advice."
)

try:
    model = load_cnn_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", type="primary"):
            resized_image = image.resize(IMG_SIZE)
            image_array = np.asarray(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # The uploaded CNN model includes a Rescaling(1./255) layer,
            # so do not divide the image by 255 again here.
            prediction = model.predict(image_array, verbose=0)
            probability = float(np.asarray(prediction).reshape(-1)[0])

            # The model uses one sigmoid output with binary cross-entropy.
            predicted_index = 1 if probability >= 0.5 else 0
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = probability if predicted_index == 1 else 1.0 - probability

            st.subheader("Prediction")
            st.write(f"**Predicted class:** {predicted_class.title()}")
            st.write(f"**Model score for malignant class:** {probability:.2%}")
            st.write(f"**Confidence in predicted class:** {confidence:.2%}")

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
