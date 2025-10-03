# app.py - Minimal working version for Streamlit Cloud
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import time
import os

# Page configuration
st.set_page_config(
    page_title="Material Classification",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Material Classification System")
st.markdown("Classify materials: **Cardboard, Glass, Metal, Paper, Plastic, Trash**")

@st.cache_resource
def load_model():
    """Load ONNX model safely"""
    try:
        # Check for model files
        if not os.path.exists('models/model_info.json'):
            return None, None, "Model info not found"
        
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        
        model_path = model_info.get('model_path', 'models/material_classifier_efficientnet_b0_streamlit.onnx')
        if not os.path.exists(model_path):
            return None, None, f"Model file not found: {model_path}"
        
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        return session, class_names, "Success"
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def preprocess_image(image):
    """Preprocess image without torchvision"""
    try:
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize (ImageNet stats)
        img_array = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0).astype(np.float32)
        
        return img_array
        
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def main():
    # Load model
    session, class_names, status = load_model()
    
    if session is None:
        st.error(f"❌ {status}")
        st.info("Please check that all model files are uploaded correctly.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Input methods
    input_method = st.radio("Choose input method:", 
                           ["Upload Image", "Use Camera"])
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose image", 
                                       type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    elif input_method == "Use Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image)
    
    # Process image
    if image is not None:
        with st.spinner("🔍 Analyzing..."):
            start_time = time.time()
            
            # Preprocess
            image_array = preprocess_image(image)
            
            if image_array is not None:
                # Predict
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: image_array})
                predictions = outputs[0]
                
                # Get results
                probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
                predicted_idx = np.argmax(probabilities[0])
                predicted_class = class_names[predicted_idx]
                confidence = probabilities[0][predicted_idx]
                
                inference_time = (time.time() - start_time) * 1000
                
                # Display results
                st.subheader("🎯 Results")
                
                # Confidence color
                if confidence > 0.8:
                    color = "green"
                elif confidence > 0.6:
                    color = "orange" 
                else:
                    color = "red"
                
                st.markdown(f"""
                <div style='background: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid {color};'>
                    <h2 style='color: {color}; margin: 0;'>{predicted_class.upper()}</h2>
                    <p style='font-size: 1.2rem;'>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⚡ Inference Time", f"{inference_time:.1f}ms")
                with col2:
                    st.metric("📊 Throughput", f"{1000/inference_time:.1f} FPS")
                
                # Top predictions
                st.subheader("📈 Top Predictions")
                top_3 = np.argsort(probabilities[0])[-3:][::-1]
                
                for idx in top_3:
                    prob = probabilities[0][idx]
                    class_name = class_names[idx]
                    st.write(f"**{class_name.title()}**: {prob:.1%}")
                    st.progress(float(prob))

if __name__ == "__main__":
    main()
