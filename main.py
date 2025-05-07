import streamlit as st
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    checkpoint = torch.load("./checkpoint_epoch_4.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

model = load_model()

# Hàm dự đoán ảnh
def RandomImagePrediction(contents):
    img_array = Image.open(BytesIO(contents)).convert("RGB")
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = data_transforms(img_array).unsqueeze(0)
    load = DataLoader(img)

    for x in load:
        x = x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        if preds[0] == 1:
            return "🐶 Chó"
        else:
            return "🐱 Mèo"
    return "Lỗi"

# Giao diện Streamlit
st.title("📸 Phân loại Chó hoặc Mèo")
uploaded_file = st.file_uploader("Tải lên ảnh (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    prediction = RandomImagePrediction(uploaded_file.read())
    st.markdown(f"### 👉 Dự đoán: **{prediction}**")
