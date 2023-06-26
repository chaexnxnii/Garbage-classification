import gdown
import streamlit as st
import requests
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import zipfile


# 모델 다운로드
file_id = "1-6OK76mqdiwZ_uuOdcQYZYVbEfceicCs"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = 'model.zip'
gdown.download(url, model_path, quiet=False)
with zipfile.ZipFile('model.zip', 'r') as zip_ref:
    zip_ref.extractall()
## 쓰레기 인식 함수 ##
@st.cache
def load_trained_model(path):
    model = load_model(path)
    return model

def classification(model, image):
    # 예측
    image_w = 64
    image_h = 64

    labels = ['plastic','glass']

    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    prediction = model.predict(np.expand_dims(data, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_label = labels[predicted_class_index]
    return predicted_label

model = load_trained_model(model_path)

if 'user_point' not in st.session_state:
    st.session_state['user_point'] = 0

#재활용품 페이지
st.title("♻️재활용품 분리배출")
  
if st.button("반납 방법 알아보기"):
  img = Image.open('src/음료 투입.png')
  img = img.resize((256, 256))
  st.image(img)
  rounded_div = """
    <div style="background-color: #f4fbee; color: #006a34; 
    ; padding: 10px; text-align: center; border-radius: 10px;">
        음료는 아래에 있는 음료 투입구에 버려주세요 
    </div>
    """
  st.markdown(rounded_div, unsafe_allow_html=True)
  st.write("")
  img = Image.open('src/페트병 분리수거.png')
  img = img.resize((256, 256))
  st.image(img)
  rounded_div = """
    <div style="background-color: #f4fbee; color: #006a34; 
    ; padding: 10px; text-align: center; border-radius: 10px;">
        페트병은 라벨을 제거하고 최대한 압축하여 배출구 위에 올려주세요
    </div>
  """
  st.markdown(rounded_div, unsafe_allow_html=True)
  st.write("")
  img = Image.open('src/유리분리수거.png')
  img = img.resize((256, 256))
  st.image(img)
  rounded_div = """
    <div style="background-color: #f4fbee; color: #006a34; 
    ; padding: 10px; text-align: center; border-radius: 10px;">
        유리병은 라벨과 뚜껑의 재질이 다를 경우 분리해서 배출해주세요 
    </div>
    """
  st.markdown(rounded_div, unsafe_allow_html=True)    
  st.write("")
upload_file = st.file_uploader('쓰레기를 투입구 위에 올려주세요',type=['jpg', 'png', 'jpeg'])

if upload_file is not None:
  img = Image.open(upload_file)
  img = img.resize((256,256))
  st.image(img, caption='Uploaded Image.', use_column_width=True)

  predicted_label = classification(model, upload_file)
  price_dict = {'plastic': 20, 'glass': 20}
 
  if predicted_label == '확인불가':
      st.markdown("확인이 불가합니다. 올바르게 배출해주세요.")
  else:
      st.markdown(f"{predicted_label}을(를) 배출하셨습니다. {price_dict[predicted_label]}포인트가 지급되었습니다!")
      st.session_state["user_point"] += price_dict[predicted_label]
    
st.sidebar.markdown(f"현재 적립포인트는 {st.session_state['user_point']}p입니다")
