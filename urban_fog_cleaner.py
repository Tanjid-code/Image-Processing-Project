import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

st.set_page_config(layout="wide")
st.title("🌁 Urban Fog Cleaner – Dehazing Tool")

uploaded_file = st.file_uploader("Upload a foggy image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    original = Image.open(uploaded_file).convert("RGB").resize((512, 512))

    # 1. Histogram Stretching (Contrast Enhancement)
    def histogram_stretch(img):
        arr = np.array(img)
        arr_min, arr_max = arr.min(), arr.max()
        stretched = ((arr - arr_min) * (255.0 / (arr_max - arr_min))).astype(np.uint8)
        return Image.fromarray(stretched)

    # 2. Approximate Frequency Domain Sharpening (High-pass using unsharp mask)
    def frequency_sharpen(img):
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # 3. Adaptive Smoothing + Sharpening
    def adaptive_smooth_sharp(img):
        smoothed = img.filter(ImageFilter.GaussianBlur(radius=2))
        enhancer = ImageEnhance.Sharpness(smoothed)
        return enhancer.enhance(2.0)

    # Apply all enhancements
    hist_img = histogram_stretch(original)
    freq_img = frequency_sharpen(original)
    adapt_img = adaptive_smooth_sharp(original)

    # Display results
    st.subheader("🖼️ Processed Images")

    col1, col2, col3, col4 = st.columns(4)
    col1.image(original, caption="Original", use_column_width=True)
    col2.image(hist_img, caption="Histogram Stretched", use_column_width=True)
    col3.image(freq_img, caption="Frequency Sharpened", use_column_width=True)
    col4.image(adapt_img, caption="Adaptive Smooth+Sharp", use_column_width=True)

else:
    st.info("⬆️ Please upload a foggy urban image to begin.")
