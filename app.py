import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ตั้งค่าหน้าตาของเว็บ
st.set_page_config(page_title="AI Web App", layout="wide")

# ใช้ session_state เพื่อจัดการหน้า
if "page" not in st.session_state:
    st.session_state.page = "Machine Learning"

# ฟังก์ชันเปลี่ยนหน้า
def change_page(new_page):
    st.session_state.page = new_page

# แสดงปุ่มนำทาง (Navigation Buttons)
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📌 Machine Learning"):
        change_page("Machine Learning")
with col2:
    if st.button("🧠 Neural Network"):
        change_page("Neural Network")
with col3:
    if st.button("🏡 Demo Machine Learning"):
        change_page("Demo Machine Learning")
with col4:
    if st.button("🐱🐶 Demo Neural Network"):
        change_page("Demo Neural Network")

st.markdown("---")  # เส้นคั่นหน้า

# 🟢 หน้า Machine Learning
if st.session_state.page == "Machine Learning":
    st.title("📌 Machine Learning")
    st.write("เนื้อหาเกี่ยวกับ Machine Learning...")

# 🔵 หน้า Neural Network
elif st.session_state.page == "Neural Network":
    st.title("🧠 Neural Network")
    st.write("เนื้อหาเกี่ยวกับ Neural Network...")

# 🟡 หน้า Demo Machine Learning
elif st.session_state.page == "Demo Machine Learning":
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
