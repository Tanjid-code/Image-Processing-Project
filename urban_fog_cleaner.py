import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles):
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        img = images[i]
        if len(img.shape) == 2:  # grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. Load the input foggy image
image_path = 'foggy_city.jpg'  # Change this to your image
original = cv2.imread(image_path)
original = cv2.resize(original, (512, 512))

# 2. Frequency Domain Enhancement (High-Pass Filtering)
def frequency_sharpen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30  # radius of low-frequency block
    mask[crow - r:crow + r, ccol - r:ccol + r] = 0  # High-pass

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

# 3. Histogram Stretching
def histogram_stretch(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    stretched = cv2.merge([h, s, v])
    return cv2.cvtColor(stretched, cv2.COLOR_HSV2BGR)

# 4. Adaptive Smoothing and Sharpening
def adaptive_smooth_sharp(img):
    # Bilateral filtering (adaptive smoothing)
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Laplacian sharpening
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    sharp = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
    return sharp

# Apply all techniques
freq_enhanced = frequency_sharpen(original)
hist_stretched = histogram_stretch(original)
adaptive_sharp = adaptive_smooth_sharp(original)

# Display all
show_images(
    [original, freq_enhanced, hist_stretched, adaptive_sharp],
    ['Original', 'Frequency Domain Sharpened', 'Histogram Stretched', 'Adaptive Smooth + Sharp']
)
