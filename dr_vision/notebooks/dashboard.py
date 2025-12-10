import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
NUM_CLASSES = 2
CLASS_NAMES = ["No_DR", "DR"]
MODEL_PATH = "best_mobilenetv2.pth"

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Transform image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# --- Streamlit UI ---
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retina image and the MobileNetV2 model will predict whether the person has Diabetic Retinopathy or not.")

# Initialize Session State
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

model = load_model()
if model is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- LOGIC FIX START ---
    # check if the file currently uploaded is different from the last one stored
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.predicted = False
        st.session_state.last_uploaded_file = uploaded_file.name
    # --- LOGIC FIX END ---

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=500)

    # Only show predict button if not predicted yet
    if not st.session_state.predicted:
        if st.button("Predict"):
            try:
                with st.spinner("Classifying..."):
                    input_tensor = transform_image(image).to(DEVICE)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = nn.Softmax(dim=1)(outputs)
                        confidence, pred_idx = torch.max(probs, 1)
                        pred_class = CLASS_NAMES[pred_idx.item()]
                        conf_score = confidence.item() * 100

                    if pred_class == "DR":
                        message = f"This person **has signs of Diabetic Retinopathy**. (Confidence: {conf_score:.2f}%)"
                        st.error(message) # Use st.error for "Positive/Bad" news for better visibility
                    else:
                        message = f"The retina appears **healthy and shows no signs of Diabetic Retinopathy**. (Confidence: {conf_score:.2f}%)"
                        st.success(message)

                    st.session_state.predicted = True

            except Exception as e:
                st.error(f"Error processing the image: {e}")
    else:
        # Optional: Show a message that prediction is done, or a button to reset manually
        if st.button("Predict Another Image"):
            # This allows the user to reset without re-uploading if they want to run it again
            st.session_state.predicted = False
            st.rerun()

elif uploaded_file is None:
    # Reset state if file is removed using the 'x' button
    st.session_state.predicted = False
    st.session_state.last_uploaded_file = None