import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache_resource
def load_cancer_model():
    return load_model("malaria_infection_cnn_model.h5")

model = load_cancer_model()

def process_image(img):
    img = img.resize((60, 60))
    img = img.convert("L")
    img = np.array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

st.title("Malaria Infection")
st.write("Please upload an image to analyze whether it is parasitized or healthy.")

file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    image = process_image(img)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    class_names = ["Uninfected", "Parasitized"]
    st.markdown(f"<h2 style='text-align: center;'>{class_names[predicted_class]}</h2>", unsafe_allow_html=True)


