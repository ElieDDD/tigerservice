import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import numpy as np
#
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
#camera = cv2.VideoCapture(0)


st.title('Image Analyzer')
image = st.file_uploader('Upload an image file',type = ['png', 'jpg', 'jpeg'])

    # Make the image a numpy array and reshape it to the models input shape.
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
image = (image / 127.5) - 1

    # Predicts the model
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

    # Print prediction and confidence score
st.text(class_name)
st.text(confidence_score)


