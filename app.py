import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
from datetime import datetime

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
        
        # Face outline landmarks
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def get_face_landmarks(self, image):
        """Extract facial landmarks using MediaPipe."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]
    
    def create_mask(self, image, landmarks):
        """Create a mask for the face region."""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        points = []
        for idx in self.FACE_OVAL:
            point = landmarks.landmark[idx]
            points.append([int(point.x * width), int(point.y * height)])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        return mask

    def apply_effect(self, image, effect='none'):
        """Apply selected effect to the image."""
        if effect == 'none':
            return image
        elif effect == 'grayscale':
            return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif effect == 'blur':
            return cv2.GaussianBlur(image, (21, 21), 0)
        elif effect == 'sepia':
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            return cv2.transform(image, kernel)

    def swap_faces(self, source_img, target_img, effect='none', blend_amount=0.8):
        """Perform face swapping between source and target images."""
        source_landmarks = self.get_face_landmarks(source_img)
        target_landmarks = self.get_face_landmarks(target_img)
        
        if source_landmarks is None or target_landmarks is None:
            return None, "Could not detect faces in one or both images"
        
        try:
            # Get key facial points for alignment
            source_points = []
            target_points = []
            for idx in [33, 133, 362, 263]:  # Key points for alignment
                source_point = source_landmarks.landmark[idx]
                target_point = target_landmarks.landmark[idx]
                
                source_points.append([
                    int(source_point.x * source_img.shape[1]),
                    int(source_point.y * source_img.shape[0])
                ])
                target_points.append([
                    int(target_point.x * target_img.shape[1]),
                    int(target_point.y * target_img.shape[0])
                ])
            
            source_points = np.float32(source_points)
            target_points = np.float32(target_points)
            
            # Calculate transformation matrix
            transform_matrix = cv2.getPerspectiveTransform(source_points, target_points)
            
            # Warp source face
            warped_source = cv2.warpPerspective(
                source_img,
                transform_matrix,
                (target_img.shape[1], target_img.shape[0])
            )
            
            # Create and blend masks
            target_mask = self.create_mask(target_img, target_landmarks)
            mask_3channel = cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # Apply selected effect
            warped_source = self.apply_effect(warped_source, effect)
            
            # Blend images
            result = (warped_source * mask_3channel * blend_amount + 
                     target_img * (1 - mask_3channel * blend_amount)).astype(np.uint8)
            
            return result, None
            
        except Exception as e:
            return None, f"Error during face swapping: {str(e)}"

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
        api_key = st.text_input("Enter your API Key", type="password", value="demo-key")
        if st.button("Save API Key"):
            st.session_state.api_key = api_key
            st.success("API Key saved successfully!")
        
        st.divider()
        
        # Effect settings
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
    
    st.markdown("""
    Upload two images to swap faces between them. The app will detect faces automatically and perform the swap.
    """)
    
    # Initialize face swapper with API key
    face_swapper = APIFaceSwapper(st.session_state.api_key or "demo-key")
    
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
            # Process images
            with st.spinner("Processing... Please wait."):
                source_bytes = np.asarray(bytearray(source_image.read()), dtype=np.uint8)
                target_bytes = np.asarray(bytearray(target_image.read()), dtype=np.uint8)
                
                source_img = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
                target_img = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
                
                result, error = face_swapper.swap_faces(
                    source_img,
                    target_img,
                    effect=effect,
                    blend_amount=blend_amount
                )
                
                if result is not None:
                    # Convert images for display
                    source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    
                    st.success("Face swap completed!")
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
                    
                    # Download button
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
        1. Upload a source image containing the face you want to use
        2. Upload a target image where you want to place the face
        3. Adjust the settings in the sidebar:
           - Select an effect to apply
           - Adjust the blend amount
        4. Wait for processing to complete
        5. Download the result
        
        **Tips for best results:**
        - Use clear, well-lit images
        - Ensure faces are visible and front-facing
        - Use high-quality images
        """)

if __name__ == "__main__":
    main()
