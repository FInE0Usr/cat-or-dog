import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# โหลดโมเดล
model = load_model('cat_dog_classifier.h5')

# ตั้งค่าหน้าเว็บ
st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the AI will classify it!")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # แปลงภาพเป็น array และทำนาย
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    # แสดงผลลัพธ์
    if prediction[0] > 0.5:
        st.write("**Prediction:** This is a Dog! 🐶")
    else:
        st.write("**Prediction:** This is a Cat! 🐱")