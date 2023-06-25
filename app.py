import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Image Classifier")

model = load_model('/path/to/your/model')  # Load the model

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])  # Image uploader

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image here (reshape, normalize, etc.)
    # Assuming the function 'preprocess' does that
    processed_image = preprocess(image)

    # Make a prediction
    predictions = model.predict(np.array([processed_image]))
    predicted_class = np.argmax(predictions)
    st.write(f"Predicted Class: {predicted_class}")
