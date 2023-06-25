# import streamlit as st
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# st.title("Image Classifier")

# model = load_model('saved_model.pb')  # Load the model

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])  # Image uploader

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)

#     # Preprocess the image here (reshape, normalize, etc.)
#     # Assuming the function 'preprocess' does that
#     processed_image = preprocess(image)

#     # Make a prediction
#     predictions = model.predict(np.array([processed_image]))
#     predicted_class = np.argmax(predictions)
#     st.write(f"Predicted Class: {predicted_class}")



# import streamlit as st
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np


# ## 쓰레기 인식 함수 ##
# def classification(image):
#   model_path = 'saved_mode.pb'

#   model = load_model(model_path)
  
#   # 예측
#   f = image
#   image_w = 64
#   image_h = 64

#   pixels = image_h * image_w * 3
#   labels = ['plastic','glass']

#   img = Image.open(f)
#   img = img.convert("RGB")
#   img = img.resize((image_w, image_h))
#   data = np.asarray(img)

#   prediction = model.predict(np.expand_dims(data, axis=0))
#   predicted_class_index = np.argmax(prediction)
#   predicted_label = labels[predicted_class_index]
#   return predicted_label
  

# if 'point' not in st.session_state:
#   st.session_state['point'] = 0
  
# ### 앱 화면 ###  

# # 초기 세션 상태 설정
# if 'option0' not in st.session_state:
#     st.session_state.option0 = '홈 화면'
# if 'point' not in st.session_state:
#   st.session_state['point'] = 0
# if 'user_point' not in st.session_state:
#     st.session_state.user_point = 0
               
# #재활용품 페이지
# if st.session_state.option1 == '재활용품 분리배출 하러 가기':
#   st.header("♻️재활용품 분리배출")
#   if st.button("반납 방법 알아보기"):
#     img = Image.open('src/안내 사진/음료 투입.png')
#     img = img.resize((256, 256))
#     st.image(img)
#     rounded_div = """
#       <div style="background-color: #f4fbee; color: #006a34; 
#       ; padding: 10px; text-align: center; border-radius: 10px;">
#           음료는 아래에 있는 음료 투입구에 버려주세요 
#       </div>
#       """.format(st.session_state['point'])
#     st.markdown(rounded_div, unsafe_allow_html=True)
#     st.write("")
#     img = Image.open('src/안내 사진/페트병 분리수거.png')
#     img = img.resize((256, 256))
#     st.image(img)
#     rounded_div = """
#       <div style="background-color: #f4fbee; color: #006a34; 
#       ; padding: 10px; text-align: center; border-radius: 10px;">
#           페트병은 라벨을 제거하고 최대한 압축하여 배출구 위에 올려주세요
#       </div>
#       """.format(st.session_state['point'])
#     st.markdown(rounded_div, unsafe_allow_html=True)
#     st.write("")
#     img = Image.open('src/안내 사진/유리분리수거.png')
#     img = img.resize((256, 256))
#     st.image(img)
#     rounded_div = """
#       <div style="background-color: #f4fbee; color: #006a34; 
#       ; padding: 10px; text-align: center; border-radius: 10px;">
#           유리병은 라벨과 뚜껑의 재질이 다를 경우 분리해서 배출해주세요 
#       </div>
#       """.format(st.session_state['point'])
#     st.markdown(rounded_div, unsafe_allow_html=True)    
#     st.write("")
#   upload_file = st.file_uploader('쓰레기를 투입구 위에 올려주세요',type=['jpg', 'png', 'jpeg'])
#   text_placeholder = st.empty()
#   if upload_file is not None:
#     text_placeholder.text('이미지 인식을 시작합니다')
#        # 이미지 출력
#     img = Image.open(upload_file)
#     img = img.resize((256,256))
#     st.image(img)
#       # 로딩 화면
#       #with st.spinner('Wait for it...'):
#         #time.sleep(3)
#       # 이미지 인식
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.name)[1]) as temp_file:
#       img.save(temp_file.name,)
#       predicted_label = classification(temp_file.name)
#       price_dict = {'plastic': 20, 'glass': 20}
#       if predicted_label == '확인불가':
#         rounded_div = """
#         <div style="background-color: #fbeeee; color: #000000;
#       ; padding: 10px; text-align: center; border-radius: 10px;">
#           <b>확인이 불가합니다. 올바르게 배출해주세요. </b>
#       </div>
#       """.format(st.session_state['point'])
#         st.markdown(rounded_div, unsafe_allow_html=True)
#       else:
#         rounded_div = """
#       <div style="background-color: #d4fbbd; color: #006a34
#       ; padding: 10px; text-align: center; border-radius: 10px;">
#            <b>{}을(를) 배출하셨습니다. {}포인트가 지급되었습니다!</b>
#       </div>
#       """
#         st.markdown(rounded_div.format(predicted_label,price_dict[predicted_label]), unsafe_allow_html=True)
#         st.session_state["user_point"] += price_dict[predicted_label]
#         text_placeholder.empty()
          
 

# for i in range(8):
#   st.sidebar.write("")

# rounded_div = """
#   <div style="background-color: #d4fbbd; color: #006a34
#   ; padding: 10px; text-align: center; border-radius: 10px;">

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


## 쓰레기 인식 함수 ##
def classification(image):
  model_path = 'saved_mode.pb'

  model = load_model(model_path)
  
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
  

if 'user_point' not in st.session_state:
    st.session_state.user_point = 0
               
#재활용품 페이지
if st.session_state.get('option1') == '재활용품 분리배출 하러 가기':
  st.header("♻️재활용품 분리배출")
  
  if st.button("반납 방법 알아보기"):
    img = Image.open('src/안내 사진/음료 투입.png')
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
    img = Image.open('src/안내 사진/페트병 분리수거.png')
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
    img = Image.open('src/안내 사진/유리분리수거.png')
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

# #       <b> 현재 적립포인트는 {}p입니다 </b>
# #   </div>
# #   """
# # st.sidebar.markdown(rounded_div.format(st.session_state['point']), unsafe_a)
