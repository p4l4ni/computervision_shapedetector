import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Shape Detection Dashboard", layout="wide")

# ---------------- GRADIENT UI + FONTS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500&display=swap');

.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    font-family: 'Inter', sans-serif;
}

/* Main title */
.main-title {
    font-family: 'Poppins', sans-serif;
    font-size: 44px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #e5e7eb;
    margin-bottom: 40px;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(10px);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    margin-bottom: 24px;
}

/* Section headers */
.section-header {
    font-family: 'Poppins', sans-serif;
    font-size: 22px;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 12px;
}

/* Metric box */
.metric-box {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white;
    padding: 16px;
    border-radius: 14px;
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    margin-top: 20px;
}

/* Table text */
[data-testid="stDataFrame"] {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Geometric Shape Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Watershed Segmentation & Shape Labelling System</div>',
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose an image containing geometric shapes",
    type=["png", "jpg", "jpeg"]
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SHAPE CLASSIFICATION ----------------
def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if area < 300 or peri == 0:
        return None

    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    v = len(approx)
    circularity = 4 * math.pi * area / (peri * peri + 1e-6)

    if circularity > 0.85:
        return "Circle"
    if v == 3:
        return "Triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        return "Square" if 0.9 < ar < 1.1 else "Rectangle"
    if v > 4:
        return "Polygon"
    return None

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.4 * dist_transform.max(), 255, 0
    )

    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    output = img.copy()
    results = []
    idx = 1

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        shape = classify_shape(cnt)
        if shape is None:
            continue

        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output, shape, (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

        results.append({
            "ID": idx,
            "Shape": shape,
            "Area (px²)": int(area),
            "Perimeter (px)": int(peri)
        })
        idx += 1

    # ---------------- DISPLAY RESULTS ----------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Segmented & Labelled Shapes")
        st.image(output, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    df = pd.DataFrame(results)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Shape Summary</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="metric-box">Total Shapes Detected: {len(df)}</div>',
        unsafe_allow_html=True
    )
#
