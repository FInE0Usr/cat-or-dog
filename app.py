import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ตั้งค่าหน้าตาของเว็บ
st.set_page_config(page_title="AI Web App", layout="wide")

# เมนูแนวนอน (Navigation Bar)
menu = st.radio("Navigation", ["Machine Learning", "Neural Network", "Demo Machine Learning", "Demo Neural Network"], horizontal=True)

# 🟢 หน้า Machine Learning
if menu == "Machine Learning":
    st.title("📌 Machine Learning")
    st.write("เนื้อหาเกี่ยวกับ Machine Learning...")
# 🟡 หน้า Demo Machine Learning
elif menu == "Demo Machine Learning":
    try:
        model = joblib.load('house_price_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🏡 House Price Prediction")
    sqft = st.number_input("🏠 ขนาดพื้นที่ (ตร.ฟุต)", min_value=500, max_value=10000, value=1000)
    bedrooms = st.number_input("🛏 จำนวนห้องนอน", min_value=1, max_value=10, value=1)
    bathrooms = st.number_input("🛁 จำนวนห้องน้ำ", min_value=1, max_value=10, value=1)

    # เตรียมข้อมูลให้ตรงกับฟีเจอร์ของโมเดล
    model_features = model.feature_names_in_
    input_data = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    input_data.loc[0, 'GrLivArea'] = sqft
    input_data.loc[0, 'BedroomAbvGr'] = bedrooms
    input_data.loc[0, 'FullBath'] = bathrooms

    if st.button("📌 Predict Price"):
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"🏡 ราคาบ้านที่คาดการณ์: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
# 🔵 หน้า Neural Network
elif menu == "Neural Network":
    st.title("🧠 Neural Network")
    st.write("เนื้อหาเกี่ยวกับ Neural Network...")


# 🔴 หน้า Demo Neural Network (Cat vs Dog Classifier)
elif menu == "Demo Neural Network":
    try:
        model = load_model('cat_dog_classifier.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🐱🐶 Cat vs Dog Classifier")
    st.write("Upload an image of a cat or dog, and the AI will classify it!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(150, 150))
        st.image(img, caption='Uploaded Image', use_column_width=True)

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        if prediction[0] > 0.5:
            st.write("**Prediction:** This is a Dog! 🐶")
        else:
            st.write("**Prediction:** This is a Cat! 🐱")
