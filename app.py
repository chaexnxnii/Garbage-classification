import streamlit as st
from PIL import Image
import requests
import zipfile
import io
import numpy as np
import tensorflow as tf

# Session state 초기화
if 'user_point' not in st.session_state:
    st.session_state['user_point'] = 0

# 모델 다운로드
model_url = 'https://github.com/chaexnxnii/Garbage-classification/raw/main/model0.zip'
response = requests.get(model_url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall()

# 쓰레기 인식 함수
def classification(image):
    model_path = './model_1'

    model = tf.saved_model.load(model_path)

    # 예측
    f = image
    image_w = 64
    image_h = 64

    labels = ['glass','plastic']
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    data = np.expand_dims(data, axis=0)
    data = tf.constant(data, dtype=tf.float32)

    prediction = model(data)
    predicted_class_index = np.argmax(prediction)
    predicted_label = labels[predicted_class_index]
    return predicted_label

#재활용품 페이지
st.title("♻️재활용품 분리배출")
  
if st.button("반납 방법 알아보기"):
    img = Image.open('src/음료 투입.png')
    img = img.resize((256, 256))
    st.image(img)
    st.markdown("음료는 아래에 있는 음료 투입구에 버려주세요")
    img = Image.open('src/페트병 분리수거.png')
    img = img.resize((256, 256))
    st.image(img)
    st.markdown("페트병은 라벨을 제거하고 최대한 압축하여 배출구 위에 올려주세요")
    img = Image.open('src/유리분리수거.png')
    img = img.resize((256, 256))
    st.image(img)
    st.markdown("유리병은 라벨과 뚜껑의 재질이 다를 경우 분리해서 배출해주세요")
    
upload_file = st.file_uploader('쓰레기를 투입구 위에 올려주세요',type=['jpg', 'png', 'jpeg'])

if upload_file is not None:
    img = Image.open(upload_file)
    img = img.resize((256,256))
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    predicted_label = classification(upload_file)
    price_dict = {'plastic': 20, 'glass': 20}

    if predicted_label == '확인불가':
        st.markdown("확인이 불가합니다. 올바르게 배출해주세요.")
    else:
        st.markdown(f"{predicted_label}을(를) 배출하셨습니다. {price_dict[predicted_label]}포인트가 지급되었습니다!")
        st.session_state["user_point"] += price_dict[predicted_label]
    
st.sidebar.markdown(f"현재 적립포인트는 {st.session_state['user_point']}p입니다")

