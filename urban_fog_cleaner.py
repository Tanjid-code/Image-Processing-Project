import streamlit as st
import cv2
import numpy as np
from PIL import Image

def pil_to_cv(image):
    """Converts a PIL Image object to an OpenCV BGR format NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    cv_image = np.array(image)
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

def simulate_fog(image, density):
    height, width, _ = image.shape
    fog_color = np.ones_like(image, dtype=np.uint8) * 255
    depth_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        depth_map[i, :] = (height - i) / height
    density_map = density * depth_map
    alpha = np.clip(1.0 - density_map[:, :, np.newaxis], 0, 1)
    foggy_image = (image * alpha + fog_color * (1 - alpha)).astype(np.uint8)
    return foggy_image

def histogram_stretching(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val == min_val:
        stretched = gray  # Avoid division by zero
    else:
        stretched = 255 * (gray - min_val) / (max_val - min_val)
    return cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def frequency_sharpening(image, radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[max(0, crow-radius):min(rows, crow+radius), max(0, ccol-radius):min(cols, ccol+radius)] = 0
    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def adaptive_sharpening(image, amount):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def main():
    st.title("🌁 Urban Fog Cleaner (Dehazing Tool)")
    st.write(
        """
        Simulate a foggy urban environment and enhance image clarity.
        
        **Instructions:**
        1. Upload a clear or foggy urban image.
        2. Optionally simulate fog.
        3. Adjust enhancement techniques.
        """
    )
    
    st.sidebar.header("Image Upload and Settings")
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        original_image_cv = pil_to_cv(pil_image)
        
        st.sidebar.subheader("1. Fog Simulation")
        simulate_fog_cb = st.sidebar.checkbox("Simulate Fog", value=False)
        fog_density = st.sidebar.slider("Fog Density", 0.0, 1.0, 0.4, 0.1)
        
        st.sidebar.subheader("2. Dehazing Techniques")
        use_hist_stretch = st.sidebar.checkbox("Histogram Stretching", value=True)
        use_freq_sharpen = st.sidebar.checkbox("Frequency Sharpening", value=True)
        freq_radius = st.sidebar.slider("Frequency Filter Radius", 1, 50, 15)
        use_adaptive_sharpen = st.sidebar.checkbox("Adaptive Sharpening", value=True)
        adaptive_amount = st.sidebar.slider("Sharpening Amount", 0.0, 5.0, 1.0, 0.1)
        
        display_image = original_image_cv.copy()
        if simulate_fog_cb:
            display_image = simulate_fog(display_image, fog_density)
        
        st.subheader("Image Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original / Foggy Image")
            st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        processed_image = display_image.copy()
        if use_hist_stretch:
            processed_image = histogram_stretching(processed_image)
        if use_freq_sharpen:
            processed_image = frequency_sharpening(processed_image, freq_radius)
        if use_adaptive_sharpen:
            processed_image = adaptive_sharpening(processed_image, adaptive_amount)
        
        with col2:
            st.markdown("### Dehazed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
