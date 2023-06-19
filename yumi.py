import streamlit as st
import os
from fastai.vision.all import *

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "export.pkl")

# Load the model
learn_inf = load_learner(model_path)

# 定义数字标签到中文名称的映射关系
label_mapping = {
    0: "病斑",
    1: "锈病",
    2: "灰斑病",
    3: "健康"
}

st.title("玉米叶子健康判断")
st.write("请上传一张玉米叶子的照片，系统将自动识别并判定。")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    
    # 根据数字标签查找对应的中文名称
    predicted_label = label_mapping[pred_idx.item()]
    
    st.write(f"玉米叶子状态: {predicted_label}; 准确率: {probs[pred_idx.item()]:.04f}")
    