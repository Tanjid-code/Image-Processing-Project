import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Constants
IMAGE_SIZE = (512, 512)

def pil_to_cv(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

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

def histogram_stretching(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
    stretched = cv2.merge((y, cr, cb))
    return cv2.cvtColor(stretched, cv2.COLOR_YCrCb2BGR)

def adaptive_smooth_sharpen(image):
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    sharpened = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
    return sharpened

def main():
    st.set_page_config(layout="wide")
    st.title("📸 Image Enhancement with 3-Step Pipeline")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        original = pil_to_cv(pil_img)

        # Step 1: Original
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original Image", width=512)

        # Step 2: Frequency Domain Enhancement
        freq_img = frequency_domain_enhancement(original)
        st.image(cv2.cvtColor(freq_img, cv2.COLOR_BGR2RGB), caption="Step 1: Frequency Domain Enhancement", width=512)

        # Step 3: Histogram Stretching
        hist_img = histogram_stretching(freq_img)
        st.image(cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB), caption="Step 2: Histogram Stretching", width=512)

        # Step 4: Adaptive Smoothing + Sharpening
        final_img = adaptive_smooth_sharpen(hist_img)
        st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Step 3: Adaptive Smoothing + Sharpening", width=512)

        st.success("✅ Enhancement completed!")

    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
