
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("plant_disease_model.keras")

# List of class names (update according to your model's training classes)
class_names = [
    'Apple___Black_rot',
    'Apple___healthy',
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Corn___Common_rust',
    'Corn___healthy'
]

# Page settings
st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Prediction App")
st.write("Upload a leaf image to identify the disease.")

# Image uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image_pil.resize((224, 224))  # Ensure size matches model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
