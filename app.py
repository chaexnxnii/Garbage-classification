# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import io
# from datetime import datetime
# import platform
# from PIL import ImageFont, ImageDraw, Image
# import time
# import re
# import tempfile
# from matplotlib import pyplot as plt
# # import cv2

# import requests
# import uuid
# import time
# import json

# from tensorflow.keras.models import load_model

# # GitHub에서 모델 다운로드
# model_url = 'https://github.com/chaexnxnii/Garbage-classification/raw/main/src/model.hy'
# model_path = 'model.hy'

# response = requests.get(model_url)
# with open(model_path, 'wb') as f:
#     f.write(response.content)

# ## 쓰레기 인식 함수 ##
# def classification(image):
#   model_path = 'model'

#   model = load_model(model_path)
  
#   # 예측
#   f = image
#   image_w = 64
#   image_h = 64

#   pixels = image_h * image_w * 3
#   labels = ['glass','plastic']
#   img = Image.open(f)
#   img = img.convert("RGB")
#   img = img.resize((image_w, image_h))
#   data = np.asarray(img)

#   prediction = model.predict(np.expand_dims(data, axis=0))
#   predicted_class_index = np.argmax(prediction)
#   predicted_label = labels[predicted_class_index]
#   return predicted_label
    
# if 'user_point' not in st.session_state:
#     st.session_state['user_point'] = 0
# if 'point' not in st.session_state:
#     st.session_state['point'] = 0  # Initializing 'point' to zero

# # Now you can safely use the 'point' key
# some_value = st.session_state['point']




# #재활용품 페이지
# st.title("♻️재활용품 분리배출")
  
# if st.button("반납 방법 알아보기"):
#   img = Image.open('src/음료 투입.png')
#   img = img.resize((256, 256))
#   st.image(img)
#   rounded_div = """
#     <div style="background-color: #f4fbee; color: #006a34; 
#     ; padding: 10px; text-align: center; border-radius: 10px;">
#         음료는 아래에 있는 음료 투입구에 버려주세요 
#     </div>
#     """.format(st.session_state['point'])
#   st.markdown(rounded_div, unsafe_allow_html=True)
#   st.write("")
#   img = Image.open('src/페트병 분리수거.png')
#   img = img.resize((256, 256))
#   st.image(img)
#   rounded_div = """
#     <div style="background-color: #f4fbee; color: #006a34; 
#     ; padding: 10px; text-align: center; border-radius: 10px;">
#         페트병은 라벨을 제거하고 최대한 압축하여 배출구 위에 올려주세요
#     </div>
#   """.format(st.session_state['point'])
#   st.markdown(rounded_div, unsafe_allow_html=True)
#   st.write("")
#   img = Image.open('src/유리분리수거.png')
#   img = img.resize((256, 256))
#   st.image(img)
#   rounded_div = """
#     <div style="background-color: #f4fbee; color: #006a34; 
#     ; padding: 10px; text-align: center; border-radius: 10px;">
#         유리병은 라벨과 뚜껑의 재질이 다를 경우 분리해서 배출해주세요 
#     </div>
#     """.format(st.session_state['point'])
#   st.markdown(rounded_div, unsafe_allow_html=True)    
#   st.write("")
# upload_file = st.file_uploader('쓰레기를 투입구 위에 올려주세요',type=['jpg', 'png', 'jpeg'])

# if upload_file is not None:
#   img = Image.open(upload_file)
#   img = img.resize((256,256))
#   st.image(img, caption='Uploaded Image.', use_column_width=True)

#   predicted_label = classification(upload_file)
#   price_dict = {'plastic': 20, 'glass': 20}
 
#   if predicted_label == '확인불가':
#       st.markdown("확인이 불가합니다. 올바르게 배출해주세요.")
#   else:
#       st.markdown(f"{predicted_label}을(를) 배출하셨습니다. {price_dict[predicted_label]}포인트가 지급되었습니다!")
#       st.session_state["user_point"] += price_dict[predicted_label]
    
# st.sidebar.markdown(f"현재 적립포인트는 {st.session_state['user_point']}p입니다")
#  #       <b> 현재 적립포인트는 {}p입니다 </b>
#  #   </div>
# # #   """
# # # st.sidebar.markdown(rounded_div.format(st.session_state['point']), unsafe_a)

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime
import platform
from PIL import ImageFont, ImageDraw, Image
import time
import re
import tempfile
from matplotlib import pyplot as plt
# import cv2

import requests
import uuid
import time
import json

import tensorflow as tf

# 모델 다운로드
model_url = 'https://github.com/chaexnxnii/Garbage-classification/raw/main/model/saved_model.zip'
model_path = 'saved_model.zip'

response = requests.get(model_url)
with open(model_path, 'wb') as f:
    f.write(response.content)

import zipfile

# 모델 압축 해제
with zipfile.ZipFile(model_path, 'r') as zip_ref:
    zip_ref.extractall('saved_model')

# 쓰레기 인식 함수
def classification(image):
  model_path = 'saved_model'

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

if 'user_point' not in st.session_state:
    st.session_state['user_point'] = 0
if 'point' not in st.session_state:
    st.session_state['point'] = 0  # Initializing 'point' to zero

# Now you can safely use the 'point' key
some_value = st.session_state['point']

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
    """.format(st.session_state['point'])
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
  """.format(st.session_state['point'])
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
    """.format(st.session_state['point'])
  st.markdown(rounded_div, unsafe_allow_html=True)    
  st.write("")
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
 #       <b> 현재 적립포인트는 {}p입니다 </b>
 #   </div>
# #   """
# # st.sidebar.markdown(rounded_div.format(st.session_state['point']), unsafe_a)
  
