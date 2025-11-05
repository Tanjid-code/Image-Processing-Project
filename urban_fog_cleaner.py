import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Constants
IMAGE_SIZE = (512, 512)

# Utility: Convert PIL image to OpenCV format and resize
def pil_to_cv(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

# Utility: Convert OpenCV image to bytes for download
def convert_to_bytes(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Filter 1: Frequency Domain Enhancement
def frequency_domain_enhancement(image, radius=70):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

# Filter 2: Histogram Stretching (CLAHE)
def histogram_stretching(image, clip=4.0):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    stretched = cv2.merge((y_clahe, cr, cb))
    return cv2.cvtColor(stretched, cv2.COLOR_YCrCb2BGR)

# Filter 3: Adaptive Smoothing + Sharpening
def adaptive_smooth_sharpen(image, sigma=75, strength=2.0):
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    sharpened = cv2.addWeighted(image, strength, smoothed, -1.0, 0)
    return sharpened

# Main App
def main():
    st.set_page_config(layout="wide")
    st.title("🌁 Urban Fog Cleaner – Image Enhancement with Adjustable Filters")

    uploaded_file = st.file_uploader("📤 Upload a foggy or low-visibility urban image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        original = pil_to_cv(pil_img)

        # === Sidebar Controls ===
        st.sidebar.header("🧩 Filter Adjustment Controls")

        freq_radius = st.sidebar.slider("Frequency Filter Radius", 10, 150, 70)
        clahe_clip = st.sidebar.slider("CLAHE Contrast (Clip Limit)", 1.0, 10.0, 4.0)
        smooth_sigma = st.sidebar.slider("Smoothing Level (Sigma)", 10, 150, 75)
        sharp_strength = st.sidebar.slider("Sharpen Strength", 0.5, 3.0, 2.0)

        st.sidebar.markdown("---")
        st.sidebar.info("Adjust sliders to fine-tune the enhancement process.")

        # Apply filters independently
        freq_only = frequency_domain_enhancement(original, radius=freq_radius)
        hist_only = histogram_stretching(original, clip=clahe_clip)
        sharp_only = adaptive_smooth_sharpen(original, sigma=smooth_sigma, strength=sharp_strength)

        # Sequential pipeline: Sharpen → Histogram → Frequency (reordered per request)
        step1 = adaptive_smooth_sharpen(original, sigma=smooth_sigma, strength=sharp_strength)  # Step 1: Sharpening
        step2 = histogram_stretching(step1, clip=clahe_clip)                                    # Step 2: Histogram Stretching
        step3 = frequency_domain_enhancement(step2, radius=freq_radius)                         # Step 3: Frequency Enhancement

        ### === DISPLAY: Individual Filters ===
        st.markdown("## Individual Filters (Applied Separately on Original)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(freq_only, cv2.COLOR_BGR2RGB), caption="Freq. Domain Only", use_column_width=True)
        with col3:
            st.image(cv2.cvtColor(hist_only, cv2.COLOR_BGR2RGB), caption="Histogram Stretch Only", use_column_width=True)
        with col4:
            st.image(cv2.cvtColor(sharp_only, cv2.COLOR_BGR2RGB), caption="Smoothing + Sharpen Only", use_column_width=True)

        ### === DISPLAY: Enhancement Pipeline ===
        st.markdown("---")
        st.markdown("## Step-by-Step Enhancement Pipeline (Sharpen → Histogram → Frequency)")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        with col6:
            st.image(cv2.cvtColor(step1, cv2.COLOR_BGR2RGB), caption="Step 1: Sharpening", use_column_width=True)
        with col7:
            st.image(cv2.cvtColor(step2, cv2.COLOR_BGR2RGB), caption="Step 2: Histogram Stretching", use_column_width=True)
        with col8:
            st.image(cv2.cvtColor(step3, cv2.COLOR_BGR2RGB), caption="Step 3: Frequency Enhancement", use_column_width=True)

        # === Download Enhanced Image ===
        st.markdown("---")
        st.markdown("### 💾 Download Final Enhanced Image")
        final_bytes = convert_to_bytes(step3)
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=final_bytes,
            file_name="enhanced_image.png",
            mime="image/png"
        )

        st.success("Enhancement complete. You can now adjust filters or download the final output.")

    else:
        st.info("Upload a foggy or low-contrast urban image to begin.")

if __name__ == "__main__":
    main()
