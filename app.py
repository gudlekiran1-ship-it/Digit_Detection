import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload a 28x28 or any handwritten digit image below to predict the digit!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.markdown(f"### 🧠 Predicted Digit: **{predicted_label}**")
