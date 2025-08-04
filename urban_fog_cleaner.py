import sys
import subprocess
import streamlit as st

def check_and_install_dependencies():
    """
    Checks if required libraries are installed and installs them if they are not.
    """
    required_packages = ["opencv-python", "numpy", "Pillow"]
    
    # Check for each package
    for package in required_packages:
        try:
            __import__(package.replace("-", "_")) # Import the package
        except ImportError:
            with st.spinner(f"Installing missing dependency: {package}... This may take a moment."):
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to install {package}. Please install it manually using: pip install {package}")
                    st.stop() # Stop the app if installation fails
    st.experimental_rerun()


# A simple way to trigger the check and installation
if "cv2" not in sys.modules or "numpy" not in sys.modules or "PIL" not in sys.modules:
    check_and_install_dependencies()

# Now that we are sure the libraries are installed, we can import them
try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    st.error("There was an issue importing necessary libraries. Please try rerunning the app.")
    st.stop()


# Helper function to convert PIL Image to NumPy array (and BGR for OpenCV)
def pil_to_cv(image):
    """Converts a PIL Image object to an OpenCV BGR format NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    cv_image = np.array(image)
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

# --- Image Processing Functions ---

def simulate_fog(image, density):
    """
    Simulates a fog effect on an image.
    
    Args:
        image (np.array): The input image (BGR format).
        density (float): The intensity of the fog, from 0.0 to 1.0.
        
    Returns:
        np.array: The image with simulated fog.
    """
    height, width, _ = image.shape
    fog_color = np.ones_like(image, dtype=np.uint8) * 255  # White fog
    
    # Create a depth map simulation (linear gradient from bottom to top)
    depth_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        depth_map[i, :] = (height - i) / height
    
    # Increase the density for a more visible effect
    density_map = density * depth_map
    
    # Blend the fog with the image based on the depth map
    alpha = np.clip(1.0 - density_map[:, :, np.newaxis], 0, 1)
    foggy_image = (image * alpha + fog_color * (1 - alpha)).astype(np.uint8)
    
    return foggy_image

def histogram_stretching(image):
    """
    Performs histogram stretching to enhance the contrast of an image.
    This is useful for images with a narrow range of pixel values, which is
    common in foggy photos.
    
    Args:
        image (np.array): The input image (BGR format).
        
    Returns:
        np.array: The image with contrast enhanced.
    """
    # Convert to grayscale to work with a single channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate min and max pixel values
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    # Apply the stretching formula
    stretched = 255 * (gray - min_val) / (max_val - min_val)
    
    # Convert back to BGR and return
    return cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def frequency_sharpening(image, radius):
    """
    Enhances edges using a high-pass filter in the frequency domain (FFT).
    
    Args:
        image (np.array): The input image (BGR format).
        radius (int): The radius of the high-pass filter. A smaller radius
                      means more sharpening (more high-frequency details).
                      
    Returns:
        np.array: The image with sharpened edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform 2D Fast Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a high-pass filter mask
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
    
    # Apply the mask
    fshift_masked = fshift * mask
    
    # Inverse FFT to get back the image
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the result and convert back to BGR
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def adaptive_sharpening(image, amount):
    """
    Applies sharpening using an unsharp masking technique.
    This method is adaptive in that it applies more sharpening to areas with
    more detail (high contrast).
    
    Args:
        image (np.array): The input image (BGR format).
        amount (float): The weight of the sharpening effect, from 0.0 to 5.0.
                        
    Returns:
        np.array: The image with adaptive sharpening applied.
    """
    # Blur the image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # Subtract the blurred image from the original to get the detail mask
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened

def main():
    """Main function to run the Streamlit application."""
    
    # --- Streamlit UI Setup ---
    st.title("Urban Fog Cleaner (Dehazing Tool)")
    st.write(
        """
        Simulate a foggy urban environment and use image enhancement techniques
        to make structures like buildings and roads more visible.
        
        **Instructions:**
        1. Upload a clear urban image or a real foggy image.
        2. (Optional) Use the sidebar to simulate fog.
        3. Select and adjust the enhancement techniques in the sidebar.
        """
    )
    
    st.sidebar.header("Image Upload and Settings")
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read the image and convert to OpenCV format
        pil_image = Image.open(uploaded_file)
        original_image_cv = pil_to_cv(pil_image)
        
        # --- Sidebar Controls ---
        st.sidebar.subheader("1. Fog Simulation")
        simulate_fog_cb = st.sidebar.checkbox(
            "Simulate Fog", value=False, help="Adds a fog effect to the uploaded image."
        )
        fog_density = st.sidebar.slider(
            "Fog Density", 0.0, 1.0, 0.4, 0.1, 
            help="Adjust the intensity of the simulated fog."
        )
        
        st.sidebar.subheader("2. Dehazing Techniques")
        use_hist_stretch = st.sidebar.checkbox(
            "Histogram Stretching", value=True,
            help="Increases contrast by stretching the range of pixel values."
        )
        
        use_freq_sharpen = st.sidebar.checkbox(
            "Frequency Sharpening", value=True,
            help="Sharpens edges using a high-pass filter in the frequency domain."
        )
        freq_radius = st.sidebar.slider(
            "Frequency Filter Radius", 1, 50, 15, 1,
            help="Smaller radius = stronger sharpening effect."
        )
        
        use_adaptive_sharpen = st.sidebar.checkbox(
            "Adaptive Sharpening", value=True,
            help="Applies sharpening by enhancing local details."
        )
        adaptive_amount = st.sidebar.slider(
            "Sharpening Amount", 0.0, 5.0, 1.0, 0.1,
            help="Adjust the intensity of the sharpening."
        )
        
        # --- Image Processing Pipeline ---
        display_image = original_image_cv.copy()
        
        # 1. Apply fog simulation if selected
        if simulate_fog_cb:
            display_image = simulate_fog(display_image, fog_density)
        
        # Create a container for the images
        st.subheader("Image Comparison")
        col1, col2 = st.columns(2)
        
        # Display the original or foggy image
        with col1:
            st.markdown("### Original / Foggy Image")
            st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # 2. Apply enhancement techniques
        processed_image = display_image.copy()
        
        if use_hist_stretch:
            processed_image = histogram_stretching(processed_image)
        
        if use_freq_sharpen:
            processed_image = frequency_sharpening(processed_image, freq_radius)
        
        if use_adaptive_sharpen:
            processed_image = adaptive_sharpening(processed_image, adaptive_amount)
            
        # Display the dehazed image
        with col2:
            st.markdown("### Dehazed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
