# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import requests
import json
from datetime import datetime

# Install required packages if not already installed
import subprocess
import sys

def install_packages():
    packages = ['mediapipe', 'opencv-python-headless', 'numpy', 'pillow', 'requests']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

install_packages()

class APIFaceSwapper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Rest of the FaceSwapper implementation remains the same
        # ... [Previous FaceSwapper code] ...

def initialize_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None

def main():
    st.set_page_config(
        page_title="AI Face Swap App",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("🎭 AI Face Swap")
    
    # API Key Configuration
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input("Enter your API Key", type="password")
        if st.button("Save API Key"):
            st.session_state.api_key = api_key
            st.success("API Key saved successfully!")
        
        if st.session_state.api_key:
            st.success("API Key is configured")
        else:
            st.warning("Please configure your API Key")
        
        st.divider()
        
        # Effect settings (only shown if API key is configured)
        if st.session_state.api_key:
            st.header("Settings")
            effect = st.selectbox(
                "Select effect",
                ['none', 'grayscale', 'blur', 'sepia'],
                help="Apply an effect to the swapped face"
            )
            
            blend_amount = st.slider(
                "Blend Amount",
                0.0, 1.0, 0.8,
                help="Adjust how strongly the face is blended"
            )
    
    # Main content
    if not st.session_state.api_key:
        st.warning("Please configure your API Key in the sidebar to use the application.")
        
        st.markdown("""
        ### How to get an API Key:
        1. Sign up at [API Provider Website]
        2. Navigate to your dashboard
        3. Generate a new API key
        4. Copy and paste the key in the sidebar
        
        ### API Key Features:
        - Secure face swapping
        - Advanced effects
        - High-quality processing
        - Usage analytics
        """)
        return
    
    st.markdown("""
    Upload two images to swap faces between them. The app will detect faces automatically and perform the swap.
    """)
    
    # Initialize face swapper with API key
    face_swapper = APIFaceSwapper(st.session_state.api_key)
    
    # Image upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Image (Face to use)")
        source_image = st.file_uploader("Upload source image", type=['jpg', 'jpeg', 'png'])
        
    with col2:
        st.subheader("Target Image (Where to place the face)")
        target_image = st.file_uploader("Upload target image", type=['jpg', 'jpeg', 'png'])
    
    if source_image and target_image:
        try:
            # Process images with API key authentication
            with st.spinner("Processing... Please wait."):
                # Your API processing code here
                # Example:
                # result = process_with_api(source_image, target_image, st.session_state.api_key)
                
                # For now, using local processing
                source_bytes = np.asarray(bytearray(source_image.read()), dtype=np.uint8)
                target_bytes = np.asarray(bytearray(target_image.read()), dtype=np.uint8)
                
                source_img = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
                target_img = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
                
                result, error = face_swapper.swap_faces(
                    source_img,
                    target_img,
                    effect=effect if 'effect' in locals() else 'none',
                    blend_amount=blend_amount if 'blend_amount' in locals() else 0.8
                )
                
                if result is not None:
                    # Display results
                    st.success("Face swap completed!")
                    
                    # Convert and display images
                    source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.subheader("Source")
                        st.image(source_rgb, use_column_width=True)
                    with cols[1]:
                        st.subheader("Target")
                        st.image(target_rgb, use_column_width=True)
                    with cols[2]:
                        st.subheader("Result")
                        st.image(result_rgb, use_column_width=True)
                    
                    # Download option
                    result_pil = Image.fromarray(result_rgb)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        result_pil.save(tmp_file.name)
                        with open(tmp_file.name, 'rb') as file:
                            st.download_button(
                                label="Download result",
                                data=file,
                                file_name=f"face_swapped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        Path(tmp_file.name).unlink()
                else:
                    st.error(error)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure both images contain clear, visible faces.")
    
    # Usage instructions
    with st.expander("How to use"):
        st.markdown("""
        1. Configure your API Key in the sidebar
        2. Upload a source image containing the face you want to use
        3. Upload a target image where you want to place the face
        4. Adjust the settings in the sidebar:
           - Select an effect to apply
           - Adjust the blend amount
        5. Wait for processing to complete
        6. Download the result
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Ensure faces are visible and front-facing
        - Use high-quality images
        - Keep API key secure
        """)

    # API Usage Stats (if available)
    if st.session_state.api_key:
        with st.expander("API Usage Statistics"):
            st.markdown("""
            - Requests today: 0/100
            - Total requests: 0
            - API Status: Active
            """)
            # You can replace these with actual API usage statistics

if __name__ == "__main__":
    main()
