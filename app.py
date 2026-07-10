import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


CLASS_NAMES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]


@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("plant_model_demo.pth", map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


model, device = load_model()

transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)


def predict_image(image: Image.Image, threshold: float):
    image_np = np.array(image.convert("RGB"))
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    predicted_labels = [
        CLASS_NAMES[i]
        for i, probability in enumerate(probabilities)
        if probability >= threshold
    ]

    return predicted_labels, probabilities


st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="centered",
)

st.title("Plant Disease Classification")
st.write(
    "Upload a leaf image and get multi-label disease predictions from the "
    "trained EfficientNet-B0 model."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
)

threshold = st.slider(
    "Prediction threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    predicted_labels, probabilities = predict_image(image, threshold)

    st.subheader("Prediction")
    if predicted_labels:
        st.success(", ".join(predicted_labels))
    else:
        st.warning("No class passed the selected threshold.")

    st.subheader("Class probabilities")
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        st.write(f"{class_name}: {probability:.3f}")
        st.progress(float(probability))
