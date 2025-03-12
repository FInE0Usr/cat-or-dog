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
    if st.button("🤖 Machine Learning"):
        change_page("Machine Learning")
with col2:
    if st.button("🏡 Demo Machine Learning 🇺🇸"):
        change_page("Demo Machine Learning")
with col3:
    if st.button("🧠 Neural Network"):
        change_page("Neural Network")
with col4:
    if st.button("🐱🐶 Demo Neural Network"):
        change_page("Demo Neural Network")

st.markdown("---")  # เส้นคั่นหน้า

# 🟢 หน้า Machine Learning
if st.session_state.page == "Machine Learning":
    st.title("🤖 Machine Learning")
    st.write("เนื้อหาเกี่ยวกับ Machine Learning...")
# 🟡 หน้า Demo Machine Learning
elif st.session_state.page == "Demo Machine Learning":
    try:
        model = joblib.load('house_price_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🏡 House Price Prediction in us 🇺🇸 ")
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
elif st.session_state.page == "Neural Network":
    st.title("🧠 Neural Network")
    st.write("ขั้นตอนแรกได้ไปหาdata set บนเว็พ kiggle และค้าหาที่ฮิดที่สุด เลยเจอ dataset cat or dogเลยคิดว่าแยกหมากับแมวก็ดูไม่ได้ง่ายขนาดนั้นเพราะมีค่อนข้างหลายสายพันธ์และค่อนข้างต่างกันมาก https://www.kaggle.com/datasets/tongpython/cat-and-dog")
    st.image("https://img5.pic.in.th/file/secure-sv1/1c2f764fab9cfb22a.png", width=600, use_container_width=False)
    st.write("และขั้นต่อมาทำการอัพ api ของ kiggle ขึ้น google colab ทำให้โหลดไฟร์zib เข้าได้ไวขึ้น และทำการแตกไฟร์ต่อ")
    st.image("https://img5.pic.in.th/file/secure-sv1/27d9df771fce3f3ad.png", width=600, use_container_width=False)
    st.write("ใช้ ImageDataGenerator จาก TensorFlow เพื่อเตรียมข้อมูลภาพสำหรับการฝึกโมเดล Machine Learning โดยเฉพาะการจำแนกภาพ (Image Classification) แบบ Binary Classification")
    st.write("สร้างโมเดล Convolutional Neural Network (CNN) สำหรับงาน Binary Classification ")
    st.write("3 ชั้น สำหรับดึง Feature จากภาพ")
    st.write("ลดขนาด Feature Map")
    st.write("2 ชั้น สำหรับการจำแนกข้อมูล")
    st.write("ใช้ Sigmoid Activation เพื่อให้ผลลัพธ์เป็นความน่าจะเป็น (0 หรือ 1)ทำให้โมเดลนี้พร้อมสำหรับการฝึกด้วยข้อมูลภาพที่เตรียมไว้")
    st.write("โค้ดนี้ใช้สำหรับ ฝึกโมเดล (Training model)")
    st.write("codeจะplotกราฟจะช่วยให้เห็นการเปลี่ยนแปลงของ Loss และ Accuracy ระหว่างการฝึก และตรวจสอบว่าโมเดล Overfitting หรือไม่")
    st.write("จากรูปโมเดลค่อนข้างเรียนรู้ได้ดี ไม่ overfittingจนเกินไป")
    st.write("codeนี้ใช้ลองทดสอบว่าทำงานได้ไหมแยกหมาแมวเบื่องต้นได้ไหม")

# 🔴 หน้า Demo Neural Network (Cat vs Dog Classifier)
elif st.session_state.page == "Demo Neural Network":
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
