import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# ✅ Paths
MODEL_PATH = 'hand_gesture_resnet.pth'
CLASS_NAMES_PATH = 'class_names.json'
IMAGE_SIZE = 224  # must match training

# ✅ Load class names
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

# ✅ Same consistent model class
class TransferResNet(nn.Module):
    def __init__(self, num_classes):
        super(TransferResNet, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ✅ Load model
@st.cache_resource
def load_model():
    num_classes = len(CLASS_NAMES)
    model = TransferResNet(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ✅ Streamlit UI
st.title("🤚 Hand Gesture Recognition")

uploaded_file = st.file_uploader("Upload a hand gesture image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict Gesture"):
        model = load_model()
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probs, 3)

            st.write("### 🔍 Top 3 Predictions:")
            for i in range(3):
                st.write(f"{CLASS_NAMES[top3_indices[0][i]]}: {top3_probs[0][i].item()*100:.2f}%")
