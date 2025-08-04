import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Constants
STANDARD_SIZE = (512, 512)

def pil_to_cv(image):
    image = image.convert("RGB")
    image = image.resize(STANDARD_SIZE)
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

def label_image_cv(image, label):
    labeled = image.copy()
    cv2.putText(labeled, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return labeled

def frequency_domain_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = 20

    # Create high-pass mask
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
    return enhanced

def histogram_stretching(image):
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Histogram stretching on Y channel
    y = cv2.normalize(y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    ycrcb = cv2.merge((y, cr, cb))
    stretched = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return stretched

def adaptive_smooth_sharpen(image):
    # Apply bilateral filter for adaptive smoothing
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # Unsharp mask: sharpen = original + alpha * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
    return sharpened

def main():
    st.set_page_config(layout="wide")
    st.title("🧠 Image Enhancement Pipeline")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        original = pil_to_cv(pil_img)

        # Step 0: Show original
        labeled_original = label_image_cv(original, "Original Image")
        st.image(cv2.cvtColor(labeled_original, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        # Step 1: Frequency Domain Enhancement
        freq_img = frequency_domain_enhancement(original)
        labeled_freq = label_image_cv(freq_img, "Frequency Domain Enhancement")
        st.image(cv2.cvtColor(labeled_freq, cv2.COLOR_BGR2RGB), caption="Step 1: Frequency Domain", use_column_width=True)

        # Step 2: Histogram Stretching
        hist_img = histogram_stretching(freq_img)
        labeled_hist = label_image_cv(hist_img, "Histogram Stretching")
        st.image(cv2.cvtColor(labeled_hist, cv2.COLOR_BGR2RGB), caption="Step 2: Histogram Stretching", use_column_width=True)

        # Step 3: Adaptive Smoothing + Sharpening
        final_img = adaptive_smooth_sharpen(hist_img)
        labeled_final = label_image_cv(final_img, "Adaptive Smoothing + Sharpening")
        st.image(cv2.cvtColor(labeled_final, cv2.COLOR_BGR2RGB), caption="Step 3: Adaptive Smooth + Sharpen", use_column_width=True)

        st.success("✅ Enhancement completed step-by-step!")

    else:
        st.info("Please upload an image to begin processing.")

if __name__ == "__main__":
    main()
