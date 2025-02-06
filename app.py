import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
model_path = "trained_plant_disease_model.keras"
def model_prediction(image_file):
    try:
        model = tf.keras.models.load_model(model_path)
        
        image = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) / 255.0  
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        return f"Error in prediction: {str(e)}"

st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])


img = Image.open("Disease.png")
st.image(img)

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.markdown("<h2 style='text-align: center;'>Upload an Image for Disease Recognition</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        
        if st.button("Predict"):
            st.snow()
            result_index = model_prediction("temp_image.jpg")
            
            class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
            
            if isinstance(result_index, int) and 0 <= result_index < len(class_name):
                st.success(f"Model is predicting it's a {class_name[result_index]}")
            else:
                st.error("Prediction error. Please check the model or input image.")
        os.remove("temp_image.jpg")

file_id = "19VmpmGdNygUYnDnDlXvVfSO_2mnGeTLO"
url = '"https://drive.google.com/file/d/19VmpmGdNygUYnDnDlXvVfSO_2mnGeTLO/view?usp=sharing"'
model_path = "trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
