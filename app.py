# streamlit_app.py - Complete Material Classification App
import streamlit as st
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
import json
import time
import os
import cv2

# Page configuration
st.set_page_config(
    page_title="Material Classification System",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border-left: 5px solid #1f77b4;
    }
    .high-confidence { color: #00cc00; font-weight: bold; }
    .medium-confidence { color: #ff9900; font-weight: bold; }
    .low-confidence { color: #ff3333; font-weight: bold; }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load ONNX model and class names with caching"""
    try:
        # Check if model files exist
        if not os.path.exists('models/model_info.json'):
            st.error("❌ Model info file not found. Please check your model files.")
            return None, None, None
            
        # Load model info
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load class names
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        
        # Check if ONNX model exists
        model_path = model_info.get('model_path', 'models/material_classifier_efficientnet_b0_streamlit.onnx')
        if not os.path.exists(model_path):
            st.error(f"❌ ONNX model not found at: {model_path}")
            return None, None, None
        
        # Create ONNX session
        session = ort.InferenceSession(model_path)
        
        st.success("✅ Model loaded successfully!")
        return session, class_names, model_info
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def preprocess_image(image):
    """Preprocess image for model inference"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.numpy()
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def predict(session, image_array):
    """Make prediction using ONNX model"""
    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_array})
        return outputs[0]
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

def create_sample_images():
    """Create placeholder sample images if they don't exist"""
    samples_dir = 'samples'
    os.makedirs(samples_dir, exist_ok=True)
    
    sample_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for class_name in sample_classes:
        sample_path = os.path.join(samples_dir, f"{class_name}.jpg")
        if not os.path.exists(sample_path):
            # Create a simple colored image as placeholder
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (200, 200), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((50, 100), f"{class_name.upper()}", fill=(255, 255, 255))
            img.save(sample_path)

def main():
    # App header
    st.markdown('<h1 class="main-header">🔄 Material Classification System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
        Classify materials into: <strong>Cardboard, Glass, Metal, Paper, Plastic, or Trash</strong>
        </p>
        <p style='font-size: 1rem; color: #888;'>
        Powered by EfficientNet-B0 | 85.6% Accuracy | Real-time Inference
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sample images if they don't exist
    create_sample_images()
    
    # Load model
    session, class_names, model_info = load_model()
    
    if session is None:
        st.error("""
        ❌ **Model not loaded. Please ensure:**
        - Model files are in the 'models' directory
        - You have the ONNX model file
        - You have 'model_info.json' and 'class_names.json'
        """)
        
        # Show what files are available
        if os.path.exists('models'):
            st.write("📁 Files in models directory:")
            for file in os.listdir('models'):
                st.write(f"   - {file}")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        st.subheader("📷 Input Method")
        input_method = st.radio(
            "Choose how to provide image:",
            ["Upload Image", "Use Camera", "Sample Images"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("🎯 Model Information")
        if model_info:
            st.write(f"**Model:** {model_info.get('model_name', 'EfficientNet-B0')}")
            st.write(f"**Input Size:** {model_info.get('input_size', [224, 224])}")
            st.write(f"**Classes:** {len(class_names)}")
            st.write(f"**Framework:** ONNX Runtime")
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.write("""
        This system classifies materials using deep learning.
        - **Accuracy:** 85.6%
        - **Speed:** ~15ms per image
        - **Training Data:** TrashNet Dataset
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    # Image input based on selection
    image = None
    image_source = ""
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "📁 Choose an image file", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of cardboard, glass, metal, paper, plastic, or trash"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_source = f"Uploaded: {uploaded_file.name}"
    
    elif input_method == "Use Camera":
        st.info("📸 Click the button below to activate your camera")
        camera_image = st.camera_input("Take a picture for classification")
        if camera_image is not None:
            image = Image.open(camera_image)
            image_source = "Camera Capture"
    
    elif input_method == "Sample Images":
        sample_options = {
            "Cardboard Sample": "samples/cardboard.jpg",
            "Glass Sample": "samples/glass.jpg",
            "Metal Sample": "samples/metal.jpg", 
            "Paper Sample": "samples/paper.jpg",
            "Plastic Sample": "samples/plastic.jpg",
            "Trash Sample": "samples/trash.jpg"
        }
        
        selected_sample = st.selectbox("Choose sample image:", list(sample_options.keys()))
        
        sample_path = sample_options[selected_sample]
        if os.path.exists(sample_path):
            if st.button("🖼️ Load Sample Image", type="primary"):
                image = Image.open(sample_path)
                image_source = f"Sample: {selected_sample}"
        else:
            st.warning("⚠️ Sample images not found. Please add sample images to the 'samples' folder.")
    
    # Process image if available
    if image is not None:
        with col1:
            st.subheader("📸 Input Image")
            
            # Display image with info
            st.image(image, use_column_width=True, caption=image_source)
            
            # Image information
            st.markdown("""
            <div class='metric-card'>
                <strong>Image Information:</strong><br>
                Size: {} × {} pixels<br>
                Mode: {}<br>
                Format: {}
            </div>
            """.format(image.size[0], image.size[1], image.mode, image.format if hasattr(image, 'format') else 'Unknown'), 
            unsafe_allow_html=True)
        
        with col2:
            # Make prediction
            with st.spinner("🔍 Analyzing image content..."):
                start_time = time.time()
                
                # Preprocess image
                image_array = preprocess_image(image)
                
                if image_array is not None:
                    # Make prediction
                    predictions = predict(session, image_array)
                    
                    if predictions is not None:
                        # Get results using softmax
                        exp_pred = np.exp(predictions - np.max(predictions))  # Numerical stability
                        probabilities = exp_pred / np.sum(exp_pred)
                        
                        predicted_class_idx = np.argmax(probabilities[0])
                        predicted_class = class_names[predicted_class_idx]
                        confidence = probabilities[0][predicted_class_idx]
                        
                        inference_time = (time.time() - start_time) * 1000
                        
                        # Display results
                        st.subheader("🎯 Classification Result")
                        
                        # Confidence styling
                        if confidence > 0.85:
                            confidence_class = "high-confidence"
                            confidence_text = "Very High Confidence"
                            emoji = "🎯"
                        elif confidence > 0.70:
                            confidence_class = "medium-confidence" 
                            confidence_text = "High Confidence"
                            emoji = "✅"
                        else:
                            confidence_class = "low-confidence"
                            confidence_text = "Moderate Confidence" 
                            emoji = "⚠️"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 class="{confidence_class}">{emoji} {predicted_class.upper()}</h2>
                            <p style='font-size: 1.5rem; margin: 10px 0;'>Confidence: {confidence:.1%}</p>
                            <p style='color: #666; font-size: 1rem;'>{confidence_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Performance metrics
                        col_perf1, col_perf2 = st.columns(2)
                        with col_perf1:
                            st.metric("⚡ Inference Time", f"{inference_time:.1f} ms")
                        with col_perf2:
                            st.metric("📊 Throughput", f"{1000/inference_time:.1f} FPS")
                        
                        # Detailed probabilities
                        st.subheader("📈 Detailed Analysis")
                        
                        # Top 3 predictions
                        top3_indices = np.argsort(probabilities[0])[-3:][::-1]
                        
                        st.write("**Top 3 Predictions:**")
                        for i, idx in enumerate(top3_indices):
                            prob = probabilities[0][idx]
                            class_name = class_names[idx]
                            
                            col_prob, col_bar, col_pct = st.columns([2, 4, 1])
                            with col_prob:
                                st.write(f"**{class_name.title()}**")
                            with col_bar:
                                st.progress(float(prob))
                            with col_pct:
                                st.write(f"{prob:.1%}")
                        
                        # All classes table
                        with st.expander("📋 View All Class Probabilities"):
                            for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
                                st.write(f"**{class_name.title()}**: {prob:.1%}")
    
    # Footer with model information
    st.markdown("---")
    
    # Model performance section
    st.subheader("🔧 System Information")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        <div class='metric-card'>
            <strong>Model Architecture</strong><br>
            EfficientNet-B0<br>
            Transfer Learning<br>
            ONNX Optimized
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div class='metric-card'>
            <strong>Performance</strong><br>
            Accuracy: 85.6%<br>
            Inference: ~15ms<br>
            Classes: 6
        </div>
        """, unsafe_allow_html=True)
    
    with col_info3:
        st.markdown("""
        <div class='metric-card'>
            <strong>Training Data</strong><br>
            TrashNet Dataset<br>
            2,500+ Images<br>
            6 Material Types
        </div>
        """, unsafe_allow_html=True)
    
    # Technical details expander
    with st.expander("🔍 Technical Details"):
        if model_info:
            st.json(model_info)
        st.write("""
        **Technical Specifications:**
        - **Framework**: PyTorch → ONNX Runtime
        - **Input Size**: 224×224×3 (RGB)
        - **Output**: 6-class probabilities  
        - **Preprocessing**: ImageNet normalization
        - **Optimization**: ONNX graph optimization
        - **Deployment**: Streamlit + ONNX Runtime
        """)

if __name__ == "__main__":
    main()