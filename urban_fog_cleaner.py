import streamlit as st
import numpy as np
import cv2
from PIL import Image

STANDARD_SIZE = (512, 512)

def pil_to_cv(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(STANDARD_SIZE)
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

def improve_contrast(image):
    # Convert to LAB color space to enhance luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel (better local contrast)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def reduce_fog_blur(image):
    # Using bilateral filter to reduce fog-like blur while preserving edges
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered

def reveal_edges_freq_domain(image, radius=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[max(0, crow - radius):min(rows, crow + radius), max(0, ccol - radius):min(cols, ccol + radius)] = 0
    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def sharpen_image(image, amount=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def main():
    st.title("Urban Image Enhancer — Reveal Edges, Improve Contrast, Reduce Fog, Sharpen")

    uploaded_file = st.file_uploader("Upload an urban image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        img = pil_to_cv(pil_img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        # Step 1: Improve contrast
        contrast_img = improve_contrast(img)
        st.image(cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB), caption="Contrast Improved", use_column_width=True)

        # Step 2: Reduce fog-like blur
        defog_img = reduce_fog_blur(contrast_img)
        st.image(cv2.cvtColor(defog_img, cv2.COLOR_BGR2RGB), caption="Fog/Blur Reduced", use_column_width=True)

        # Step 3: Reveal edges (frequency domain)
        edges_img = reveal_edges_freq_domain(defog_img, radius=15)
        st.image(cv2.cvtColor(edges_img, cv2.COLOR_BGR2RGB), caption="Obscured Edges Revealed", use_column_width=True)

        # Step 4: Sharpen overall structure
        sharpened_img = sharpen_image(edges_img, amount=1.5)
        st.image(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB), caption="Sharpened Image", use_column_width=True)

        st.markdown("### Final Enhanced Image")
        st.image(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    else:
        st.info("Please upload an image to start processing.")

if __name__ == "__main__":
    main()
