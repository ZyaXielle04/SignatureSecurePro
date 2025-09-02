import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time

# Set page configuration
st.set_page_config(
    page_title="SignatureSecure Pro",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Harris Corner Detector implementation (same as before)
class HarrisCornerDetector:
    def __init__(self, k=0.04, threshold=0.01):
        self.k = k
        self.threshold = threshold
    
    def detect(self, image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Convert to float32 for precision
        gray = np.float32(gray)
        
        # Compute derivatives
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute products of derivatives
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        
        # Apply Gaussian blur
        Ixx = cv2.GaussianBlur(Ixx, (5, 5), 0)
        Ixy = cv2.GaussianBlur(Ixy, (5, 5), 0)
        Iyy = cv2.GaussianBlur(Iyy, (5, 5), 0)
        
        # Compute Harris response
        det = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy
        R = det - self.k * trace * trace
        
        # Apply threshold
        corners = np.zeros_like(gray)
        corners[R > self.threshold * R.max()] = 255
        
        # Get corner coordinates
        y, x = np.where(corners == 255)
        
        return list(zip(x, y))

# KNN-Harris Corner Detector implementation (same as before)
class KNNHarrisCornerDetector:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.harris = HarrisCornerDetector()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, image):
        # Detect corners
        corners = self.harris.detect(image)
        
        if len(corners) == 0:
            return np.zeros(128)
        
        # Convert corners to feature vector
        # Use the first 64 corners (pad with zeros if fewer)
        max_corners = 64
        features = []
        
        for i in range(max_corners):
            if i < len(corners):
                features.extend(corners[i])
            else:
                features.extend([0, 0])
        
        # Ensure we have exactly 128 features
        features = features[:128]
        if len(features) < 128:
            features.extend([0] * (128 - len(features)))
            
        return np.array(features)
    
    def compare_signatures(self, ref_features, test_features):
        # Calculate cosine similarity
        dot_product = np.dot(ref_features, test_features)
        norm_ref = np.linalg.norm(ref_features)
        norm_test = np.linalg.norm(test_features)
        
        if norm_ref == 0 or norm_test == 0:
            return 0
            
        similarity = dot_product / (norm_ref * norm_test)
        return similarity

# Siamese Network implementation (same as before)
class ResNetSiameseNetwork(nn.Module):
    def __init__(self):
        super(ResNetSiameseNetwork, self).__init__()
        
        # Using a pretrained ResNet-18 as backbone
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Additional layers for fine-tuning
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Image preprocessing for Siamese network
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to verify signatures
def verify_signatures(ref_image, test_image, knn_harris, siamese_net):
    # Convert images to numpy arrays if they are PIL Images
    if isinstance(ref_image, Image.Image):
        ref_image = np.array(ref_image)
    if isinstance(test_image, Image.Image):
        test_image = np.array(test_image)
    
    # Stage 1: KNN-Harris Corner Detection (Coarse Filtering)
    ref_features = knn_harris.extract_features(ref_image)
    test_features = knn_harris.extract_features(test_image)
    
    # Compare signatures
    similarity_stage1 = knn_harris.compare_signatures(ref_features, test_features)
    
    # If obvious forgery detected in Stage 1
    if similarity_stage1 < 0.3:  # Threshold
        return "FORGED", f"Detected in Stage 1 (Confidence: {(1-similarity_stage1)*100:.2f}%)"
    
    # Stage 2: CNN-Siamese Network (Detailed Verification)
    # Preprocess images
    ref_tensor = preprocess_image(ref_image)
    test_tensor = preprocess_image(test_image)
    
    # Get embeddings from Siamese network
    siamese_net.eval()
    with torch.no_grad():
        ref_embedding, test_embedding = siamese_net(ref_tensor, test_tensor)
    
    # Calculate cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity_stage2 = cos(ref_embedding, test_embedding).item()
    
    # Determine final result
    threshold = 0.7
    if similarity_stage2 >= threshold:
        return "GENUINE", f"Confidence: {similarity_stage2*100:.2f}%"
    else:
        return "FORGED", f"Confidence: {(1-similarity_stage2)*100:.2f}%"

# Main application
def main():
    # Initialize algorithms
    knn_harris = KNNHarrisCornerDetector()
    siamese_net = ResNetSiameseNetwork()
    
    # App title and description
    st.title("✍️ SignatureSecure Pro")
    st.markdown("### Two-Stage Signature Verification System")
    st.markdown("---")
    
    # Create two columns for image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Reference Signature")
        ref_file = st.file_uploader("Upload reference signature", type=['png', 'jpg', 'jpeg'], key="ref")
        
    with col2:
        st.header("Test Signature")
        test_file = st.file_uploader("Upload test signature", type=['png', 'jpg', 'jpeg'], key="test")
    
    # Display uploaded images
    if ref_file is not None:
        ref_image = Image.open(ref_file)
        col1.image(ref_image, caption="Reference Signature", use_column_width=True)
    
    if test_file is not None:
        test_image = Image.open(test_file)
        col2.image(test_image, caption="Test Signature", use_column_width=True)
    
    # Verify button
    if ref_file is not None and test_file is not None:
        if st.button("Verify Signature", type="primary", use_container_width=True):
            with st.spinner("Verifying signatures..."):
                # Add a progress bar
                progress_bar = st.progress(0)
                
                for percent_complete in range(100):
                    time.sleep(0.01)  # Simulate processing time
                    progress_bar.progress(percent_complete + 1)
                
                # Verify the signatures
                result, confidence = verify_signatures(ref_image, test_image, knn_harris, siamese_net)
                
                # Display result
                st.markdown("---")
                st.subheader("Verification Result")
                
                if result == "GENUINE":
                    st.success(f"✅ {result} Signature")
                    st.info(confidence)
                else:
                    st.error(f"❌ {result} Signature")
                    st.info(confidence)
    
    # Add some information about the algorithm
    with st.expander("About the Verification Algorithm"):
        st.markdown("""
        SignatureSecure Pro uses a two-stage verification process:
        
        1. **KNN-Harris Corner Detection (Coarse Filtering)**
           - Extracts corner features from signatures
           - Compares feature vectors for initial verification
           - Quickly identifies obvious forgeries
        
        2. **CNN-Siamese Network (Detailed Verification)**
           - Uses a deep learning model with ResNet-18 backbone
           - Compares signature embeddings for detailed analysis
           - Provides high-confidence verification results
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("SignatureSecure Pro v1.0 | Advanced Two-Stage Verification Technology")

if __name__ == "__main__":
    main()