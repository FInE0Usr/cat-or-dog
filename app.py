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
    st.title("🤖 Machine Learning: House Price Prediction")

    st.subheader("ข้อมูล Dataset ที่ใช้")
    st.write(
        """
        **Kaggle House Prices Dataset**\n
        ตอนแรกได้เริ่มทำการหาข้อมูลผ่าน ChatGPT และได้คำแนะนำเกี่ยวกับเว็บไซต์ Kaggle ค่ะ\n
        อันนี้ข้อมูลเกี่ยวกับราคาบ้านจากเมือง Ames, Iowa, USA ใช้สำหรับสร้างโมเดลพยากรณ์ราคาบ้าน  
        - data source: [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)\n
        """
    )
    st.subheader("ฟีเจอร์ที่ใช้ในโมเดล")
    st.write(
        """
        ฟีเจอร์หลักที่ใช้ในการพยากรณ์ราคาบ้าน :
        - **LotArea** (ขนาดที่ดิน, ตารางฟุต)  
        - **BedroomAbvGr** (จำนวนห้องนอน)  
        - **FullBath** (จำนวนห้องน้ำ)  
        - **SalePrice** (ราคาขายบ้าน)\n
        """
    )
    st.subheader("ขั้นตอนการเตรียมข้อมูลใน Google Colab")
    st.image("https://img2.pic.in.th/pic/Screenshot-1.jpg", width=600, use_container_width=False)
    st.write(
        """
        **อัปโหลด Datasetและแตกไฟล์ ZIP**\n
        - ใช้ `files.upload()` เพื่ออัปโหลดไฟล์ ZIP จากเครื่อง
        - ใช้ `!unzip house.zip` เพื่อแตกไฟล์\n
        """
    )
    st.image("https://img2.pic.in.th/pic/Screenshot-2aa9205adfd6055d3.jpg", width=600, use_container_width=False)
    st.write(
        """
        **โหลดข้อมูลและการตรวจสอบโครงสร้างของ Dataset**\n
        หลังจากแตกไฟล์ ZIP แล้ว เราจะใช้ `pd.read_csv()` เพื่อโหลดข้อมูลเข้าสู่ Pandas DataFrame  
        - `train.csv` → ใช้ในการฝึกโมเดล (มีราคาบ้าน)
        - `test.csv` → ใช้ในการพยากรณ์ราคาบ้าน (ไม่มีราคาบ้าน)
        - `sample_submission.csv` → ตัวอย่างไฟล์ที่ใช้ส่งผลลัพธ์ให้ Kaggle \n

        หลังจากโหลดข้อมูลแล้ว เราจะตรวจสอบโครงสร้างของ dataset ด้วย `shape` และ `head()`  
        - `train_data.shape` → ดูขนาดของข้อมูล Train  
        - `test_data.shape` → ดูขนาดของข้อมูล Test  
        - `train_data.head()` → ดูตัวอย่างข้อมูล 5 แถวแรก\n 
        
        """
    )
    st.image("https://img2.pic.in.th/pic/Screenshot-3.jpg", width=600, use_container_width=False)
    st.write(
       """
       **การจัดการข้อมูลที่ขาดหาย (Missing Data)**\n
        - ใช้ `.isnull().sum()` เพื่อตรวจสอบข้อมูลที่ขาดหาย  
        - แก้ไขข้อมูลที่ขาดหายด้วย:
        - เติมค่า `None` ให้กับข้อมูลที่เป็นข้อความ  
        - เติมค่ามัธยฐาน (Median) ให้กับข้อมูลตัวเลข\n
        
        """
    )
    st.image("https://img2.pic.in.th/pic/Screenshot-4e10b6e8e37cc2879.jpg", width=600, use_container_width=False)
    st.write(
        """
        **การแปลงข้อมูล Categorical เป็นตัวเลข (One-Hot Encoding)**\n
        เนื่องจาก dataset มีบางคอลัมน์ที่เป็นข้อความ เช่น `Neighborhood` และ `HouseStyle`  
        เราจึงต้องแปลงข้อมูลเหล่านี้เป็นตัวเลขโดยใช้ **One-Hot Encoding** ค่ะ\n
        
        """
    )
    st.image("https://img2.pic.in.th/pic/Screenshot-5.jpg", width=600, use_container_width=False)
    st.write(
        """
        **การทำให้ Train / Test มีคอลัมน์ตรงกันและการแยก Features (X) กับ Target (y)**\n
        หลังจากใช้ **One-Hot Encoding** คอลัมน์ของ `train_data` และ `test_data` อาจจะไม่เหมือนกัน  
        ก่อน Train โมเดล เราต้องแยกข้อมูลเป็น:  
        - **X** → Features ที่ใช้พยากรณ์ เช่น `GrLivArea`, `BedroomAbvGr`, `FullBath`  
        - **y** → Target ที่ต้องทำนาย (`SalePrice`)  

        **วิธีแยกข้อมูล:**  
        - ลบ `Id` เพราะไม่เกี่ยวกับการพยากรณ์  
        - ลบ `SalePrice` ออกจาก `X`  
        - `X_test` มีเฉพาะ Features เพราะไม่มี Target\n
        
        """
    ) 
    st.image("https://img2.pic.in.th/pic/Screenshot-6.jpg", width=600, use_container_width=False)
    st.write(
          """
          **การแบ่งข้อมูล Train และ Validation** \n
          แบ่งข้อมูลเป็น 80% สำหรับ Train และ 20% สำหรับ Validation  
          และใช้ `train_test_split()` เพื่อช่วยให้โมเดลเรียนรู้และทดสอบได้แม่นยำขึ้น\n  
          
            """
    )  
    st.image("https://img5.pic.in.th/file/secure-sv1/Screenshot-7a9374fee9b8f1a4d.jpg", width=600, use_container_width=False)
    st.write(
        """
        **การ Train และประเมินผลโมเดล** \n
        ใช้ **Random Forest Regressor** ในการ Train โมเดล  
        และวัดความคลาดเคลื่อนด้วย **Mean Absolute Error (MAE)**\n  
        
        """
    )
    st.image("https://img5.pic.in.th/file/secure-sv1/Screenshot-8.jpg", width=600, use_container_width=False)
    st.write(
        """
        **ตรวจสอบคอลัมน์ของ Train และ Test** \n 
        ตรวจสอบว่าคอลัมน์ใน Train และ Test ตรงกันหรือไม่ 
        - `X_train`และ`X_test` มี 287 คอลัมน์ ซึ่งหมายความว่าข้อมูลทั้งสองชุดตรงกันแล้ว\n
        
        """
    ) 
    st.image("https://img2.pic.in.th/pic/Screenshot-9d2484ac20106009f.jpg", width=600, use_container_width=False)
    st.write(
        """
        **ตรวจสอบและลบ `SalePrice` ออกจาก X_test** \n
        เนื่องจาก `X_test` ใช้สำหรับทำนายราคาบ้าน  
        เพื่อให้แน่ใจว่าไม่มีคอลัมน์ `SalePrice` อยู่ในข้อมูล\n  
        
        """
     )
    st.image("https://img5.pic.in.th/file/secure-sv1/Screenshot-10424041db1cb5a396.jpg", width=600, use_container_width=False)
    st.write(
        """
        **การสร้างไฟล์ Submission และบันทึกโมเดล** \n
        ใช้โมเดลที่เทรนเสร็จแล้วพยากรณ์ราคาบ้านและบันทึกผลลัพธ์เป็นไฟล์ CSV เพื่อใช้ส่งผลลัพธ์\n  
        
        """
    )
    st.image("https://img2.pic.in.th/pic/Screenshot-11399ea42268c8c7e8.jpg", width=600, use_container_width=False)
    st.write(
        """
        **โหลดโมเดลที่บันทึกไว้ และตรวจสอบไฟล์ Submission** \n
        - โมเดล `house_price_model.pkl` ที่เทรนไว้สามารถโหลดมาใช้งานได้ทันที\n  
        โดยไม่ต้องเทรนใหม่  

        - ต้องตรวจสอบไฟล์ `submission.csv`  
        ซึ่งเป็นผลลัพธ์การพยากรณ์ราคาบ้าน สำหรับนำไปใช้งานต่อ\n  
        
        """
)
    st.image("https://img5.pic.in.th/file/secure-sv1/Screenshot-12.jpg", width=600, use_container_width=False)
    st.write(
        """
        **ตรวจสอบจำนวนและชื่อของฟีเจอร์ที่ใช้ในโมเดล** \n
        ก่อนใช้งานโมเดล เราต้องตรวจสอบว่าจำนวนฟีเจอร์ที่ใช้ตรงกับโมเดลที่บันทึกไว้และดูว่าฟีเจอร์ที่โมเดลใช้มีอะไรบ้าง\n  
        
        """
    )
    st.image("https://img5.pic.in.th/file/secure-sv1/Screenshot-13.jpg", width=600, use_container_width=False)
    st.write(
        """
        **ทำนายราคาบ้านด้วยโมเดลที่ฝึกไว้** \n
        ใส่ค่าพื้นที่จำนวนห้องนอนและห้องน้ำจากนั้นโมเดลจะพยากรณ์ราคาบ้านให้\n  
        
        """
    )
# 🟡 หน้า Demo Machine Learning
elif st.session_state.page == "Demo Machine Learning":
    try:
        model = joblib.load('house_price_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🏡 House Price Prediction in US 🇺🇸 ")

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
    st.image("https://img2.pic.in.th/pic/33d29ee08dd6fd275.png", width=600, use_container_width=False)
    st.write("สร้างโมเดล Convolutional Neural Network (CNN) สำหรับงาน Binary Classification ")
    st.write("3 ชั้น สำหรับดึง Feature จากภาพ")
    st.write("ลดขนาด Feature Map")
    st.write("2 ชั้น สำหรับการจำแนกข้อมูล")
    st.write("ใช้ Sigmoid Activation เพื่อให้ผลลัพธ์เป็นความน่าจะเป็น (0 หรือ 1)ทำให้โมเดลนี้พร้อมสำหรับการฝึกด้วยข้อมูลภาพที่เตรียมไว้")
    st.image("https://img2.pic.in.th/pic/418afd8ea65b7e17a.png", width=600, use_container_width=False)
    st.write("โค้ดนี้ใช้สำหรับ ฝึกโมเดล (Training model)")
    st.image("https://img2.pic.in.th/pic/Screenshot-from-2025-03-12-23-45-52.png", width=600, use_container_width=False)
    st.write("codeจะplotกราฟจะช่วยให้เห็นการเปลี่ยนแปลงของ Loss และ Accuracy ระหว่างการฝึก และตรวจสอบว่าโมเดล Overfitting หรือไม่")
    st.image("https://img2.pic.in.th/pic/6491366b23e31efd5.png", width=600, use_container_width=False)
    st.write("จากรูปโมเดลค่อนข้างเรียนรู้ได้ดี ไม่ overfittingจนเกินไป")
    st.image("https://img5.pic.in.th/file/secure-sv1/810e34025f882f14e.png", width=600, use_container_width=False)
    st.write("codeนี้ใช้ลองทดสอบว่าทำงานได้ไหมแยกหมาแมวเบื่องต้นได้ไหมและ savemodel ใน .h57")

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
