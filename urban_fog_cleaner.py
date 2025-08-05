import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Constants
IMAGE_SIZE = (512, 512)

# Utility: Convert PIL image to OpenCV format and resize
def pil_to_cv(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

# Filter 1: Frequency Domain Enhancement
def frequency_domain_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = 20

    mask = np.ones((rows, cols), np.uint8)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

# Filter 2: Histogram Stretching
def histogram_stretching(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
    stretched = cv2.merge((y, cr, cb))
    return cv2.cvtColor(stretched, cv2.COLOR_YCrCb2BGR)

# Filter 3: Adaptive Smoothing + Sharpening
def adaptive_smooth_sharpen(image):
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    sharpened = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
    return sharpened

# Main App
def main():
    st.set_page_config(layout="wide")
    st.title("🌁 Urban Fog Cleaner – Image Enhancement with Filters")

    uploaded_file = st.file_uploader("📤 Upload a foggy or low-visibility urban image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        original = pil_to_cv(pil_img)

        # Apply all filters independently on original
        freq_only = frequency_domain_enhancement(original)
        hist_only = histogram_stretching(original)
        sharp_only = adaptive_smooth_sharpen(original)

        # Apply filters sequentially (pipeline)
        step1 = frequency_domain_enhancement(original)
        step2 = histogram_stretching(step1)
        step3 = adaptive_smooth_sharpen(step2)

        ### === DISPLAY: Original + Filtered Separately === ###
        st.markdown("## 🎯 Filter Effects Applied Separately on Original Image")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original (512×512)", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(freq_only, cv2.COLOR_BGR2RGB), caption="Frequency Domain Only", use_column_width=True)
        with col3:
            st.image(cv2.cvtColor(hist_only, cv2.COLOR_BGR2RGB), caption="Histogram Stretching Only", use_column_width=True)
        with col4:
            st.image(cv2.cvtColor(sharp_only, cv2.COLOR_BGR2RGB), caption="Adaptive Smoothing Only", use_column_width=True)

        ### === DISPLAY: Sequential Enhancement Steps === ###
        st.markdown("---")
        st.markdown("## 🔁 Step-by-Step Enhancement Pipeline")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        with col6:
            st.image(cv2.cvtColor(step1, cv2.COLOR_BGR2RGB), caption="Step 1: Frequency Domain", use_column_width=True)
        with col7:
            st.image(cv2.cvtColor(step2, cv2.COLOR_BGR2RGB), caption="Step 2: Histogram Stretching", use_column_width=True)
        with col8:
            st.image(cv2.cvtColor(step3, cv2.COLOR_BGR2RGB), caption="Step 3: Smoothing + Sharpening", use_column_width=True)

        st.success("✅ Enhancement complete. You can now compare individual filter effects vs the combined pipeline.")

    else:
        st.info("Upload a foggy or low-contrast urban image to begin.")

if __name__ == "__main__":
    main()
