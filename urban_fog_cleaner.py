import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide")
st.title("🌆 Urban Fog Cleaner (Dehazing Tool)")

# Image uploader
uploaded_file = st.file_uploader("Upload a foggy image (jpg, png, jpeg):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = cv2.resize(original, (512, 512))

    def frequency_sharpen(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = gray.shape
        crow, ccol = rows // 2 , cols // 2
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 30
        mask[crow - r:crow + r, ccol - r:ccol + r] = 0  # high-pass

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(img_back)

    def histogram_stretch(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        stretched = cv2.merge([h, s, v])
        return cv2.cvtColor(stretched, cv2.COLOR_HSV2BGR)

    def adaptive_smooth_sharp(img):
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        sharp = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
        return sharp

    # Process images
    freq_image = frequency_sharpen(original)
    hist_image = histogram_stretch(original)
    adapt_image = adaptive_smooth_sharp(original)

    # Display images
    st.subheader("🔍 Processed Image Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.image(original, caption="Original", use_column_width=True)
    col2.image(freq_image, caption="Frequency Sharpened", use_column_width=True, channels="GRAY")
    col3.image(hist_image, caption="Histogram Stretched", use_column_width=True)
    col4.image(adapt_image, caption="Adaptive Sharpening", use_column_width=True, channels="GRAY")
else:
    st.info("⬆️ Upload a foggy city image to start dehazing.")
