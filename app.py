import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import os
import shutil

# ================= CONFIG =================
MODEL_PATH = "ai_image_detector.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_INPUT_DIR = os.path.join(BASE_DIR, "video_input")
VIDEO_FRAMES_DIR = os.path.join(BASE_DIR, "video_frames")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ================= HELPER =================
def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return probs.cpu().numpy()  # [AI, Real]

# ================= UI =================
st.title("AI Image & Video Detector")
st.write("Detect whether an image or video is AI-generated or real.")

# ==================================================
# 🖼️ IMAGE DETECTOR
# ==================================================
st.subheader("🖼️ Image Detector (AI vs Real)")

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key="image_upload"
)

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    probs = predict_image(image)
    ai_prob = probs[0] * 100
    real_prob = probs[1] * 100

    if ai_prob > real_prob:
        st.error(f"🔴 This image is likely AI-generated ({ai_prob:.2f}%)")
    else:
        st.success(f"🟢 This image is likely REAL ({real_prob:.2f}%)")

st.markdown("---")

# ==================================================
# 🎬 VIDEO DETECTOR
# ==================================================
st.subheader("🎬 Video Detector (AI vs Real)")

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"],
    key="video_upload"
)

if video_file is not None:
    st.info("Analyzing video... please wait")

    os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_FRAMES_DIR, exist_ok=True)

    video_path = os.path.join(VIDEO_INPUT_DIR, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Video could not be opened")
        st.stop()

    frame_count = 0
    saved_frames = 0
    ai_count = 0
    real_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            cv2.imwrite(
                os.path.join(VIDEO_FRAMES_DIR, f"frame_{saved_frames}.jpg"),
                frame
        )

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs = predict_image(image)  # ← МІНЕ ОСЫ ЖОЛ ЖЕТПЕЙ ТҰРҒАН

            if probs[0] > probs[1]:
                ai_count += 1
            else:
                real_count += 1

            saved_frames += 1
        frame_count += 1

    cap.release()

    total = ai_count + real_count
    ai_percent = (ai_count / total) * 100
    real_percent = (real_count / total) * 100

    st.subheader("🎯 Video Result")
    st.write(f"AI-generated frames: {ai_count}")
    st.write(f"Real frames: {real_count}")

    if ai_percent >= 60:
        st.error(f"🔴 This video is likely AI-generated ({ai_percent:.2f}%)")
    else:
        st.success(f"🟢 This video is likely REAL ({real_percent:.2f}%)")

    st.caption("Result is based on frame-by-frame analysis")

    # shutil.rmtree(VIDEO_INPUT_DIR, ignore_errors=True)
    # shutil.rmtree(VIDEO_FRAMES_DIR, ignore_errors=True)
    