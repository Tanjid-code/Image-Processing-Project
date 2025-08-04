import streamlit as st
import numpy as np
import cv2
from PIL import Image

def pil_to_cv(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

def simulate_fog(image, density=0.4):
    height, width, _ = image.shape
    fog_color = np.ones_like(image, dtype=np.uint8) * 255
    depth_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        depth_map[i, :] = (height - i) / height
    density_map = density * depth_map
    alpha = np.clip(1.0 - density_map[:, :, np.newaxis], 0, 1)
    foggy = (image * alpha + fog_color * (1 - alpha)).astype(np.uint8)
    return foggy

def histogram_stretching(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val == min_val:
        stretched = gray
    else:
        stretched = 255 * (gray - min_val) / (max_val - min_val)
    return cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def frequency_domain_enhancement(image, radius=15):
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

def adaptive_smoothing_sharpening(image, amount=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def main():
    st.title("Urban Fog Cleaner (Dehazing Tool)")
    st.write("""
    Enhance urban images to see buildings and roads clearly through fog or haze.
    
    Upload an image, optionally simulate fog, and apply enhancement techniques:
    Frequency Domain Enhancement, Histogram Stretching, Adaptive Smoothing and Sharpening.
    """)

    uploaded_file = st.file_uploader("Upload an urban image (clear or foggy)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        orig_cv = pil_to_cv(pil_img)
        st.image(cv2.cvtColor(orig_cv, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        fog_simulate = st.checkbox("Simulate Fog on Image", value=False)
        fog_density = st.slider("Fog Density", 0.0, 1.0, 0.4) if fog_simulate else 0.0

        if fog_simulate:
            foggy_img = simulate_fog(orig_cv, fog_density)
            st.image(cv2.cvtColor(foggy_img, cv2.COLOR_BGR2RGB), caption="Simulated Fog Image", use_column_width=True)
        else:
            foggy_img = orig_cv.copy()

        # Histogram stretching
        hist_stretched = histogram_stretching(foggy_img)
        st.image(cv2.cvtColor(hist_stretched, cv2.COLOR_BGR2RGB), caption="Histogram Stretched Image", use_column_width=True)

        # Frequency domain enhancement
        freq_radius = st.slider("Frequency Domain Filter Radius", 1, 50, 15)
        freq_enhanced = frequency_domain_enhancement(hist_stretched, freq_radius)
        st.image(cv2.cvtColor(freq_enhanced, cv2.COLOR_BGR2RGB), caption="Frequency Domain Enhanced Image", use_column_width=True)

        # Adaptive smoothing and sharpening
        sharpen_amount = st.slider("Adaptive Sharpening Amount", 0.0, 5.0, 1.0)
        adapt_sharp = adaptive_smoothing_sharpening(freq_enhanced, sharpen_amount)
        st.image(cv2.cvtColor(adapt_sharp, cv2.COLOR_BGR2RGB), caption="Adaptive Smoothing and Sharpening", use_column_width=True)

        st.markdown("### Final Dehazed Image")
        st.image(cv2.cvtColor(adapt_sharp, cv2.COLOR_BGR2RGB), use_column_width=True)

    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
