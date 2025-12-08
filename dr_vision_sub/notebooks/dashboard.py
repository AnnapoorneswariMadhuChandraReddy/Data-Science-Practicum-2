# app.py

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import os

# ===============================
# Device
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Constants
# ===============================
NUM_CLASSES = 2
CLASS_NAMES = ["No_DR", "DR"]
MODEL_PATH = "best_swin_transformer.pth"  # Update if your path is different

# ===============================
# Load Swin Transformer Model
# ===============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
if model is None:
    st.stop()

# ===============================
# Transform Image
# ===============================
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retina image and the model will predict DR or No DR.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        input_tensor = transform_image(image).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = nn.Softmax(dim=1)(outputs)
            confidence, pred_idx = torch.max(probs, 1)
            pred_class = CLASS_NAMES[pred_idx.item()]
            conf_score = confidence.item() * 100

        st.success(f"Prediction: **{pred_class}** ({conf_score:.2f}% confidence)")
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")
