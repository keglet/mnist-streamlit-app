import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load checkpoint 13 MNIST CNN model
model = tf.keras.models.load_model('mnist_model.keras')

# MNIST digit class names
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a handwritten digit image (0-9) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image preview
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(image, caption='Uploaded Image', width=150)

    # Preprocess image to match checkpoint 13 model input
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    st.success(f"Predicted Digit: **{class_names[predicted_class]}**")
    st.info(f"Confidence: **{round(confidence, 2)}%**")

    # Probability bar chart
    st.subheader("Prediction Probabilities")
    prob_dict = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)