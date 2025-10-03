# app.py - Fixed ONNX Runtime version
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
    """Load ONNX model with explicit providers"""
    
    # Check what files exist
    st.sidebar.subheader("📁 File Status")
    
    files_to_check = [
        'models/model_info.json',
        'models/class_names.json', 
        'models/material_classifier_efficientnet_b0_streamlit.onnx'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            st.sidebar.success(f"✅ {file_path}")
        else:
            st.sidebar.error(f"❌ {file_path}")
    
    try:
        # Check if model files exist
        if not os.path.exists('models/model_info.json'):
            return None, None, "❌ models/model_info.json not found"
        
        if not os.path.exists('models/class_names.json'):
            return None, None, "❌ models/class_names.json not found"
            
        # Load model info
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Load class names
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        
        # Check if ONNX model exists
        model_path = model_info.get('model_path', 'models/material_classifier_efficientnet_b0_streamlit.onnx')
        if not os.path.exists(model_path):
            return None, None, f"❌ ONNX model not found: {model_path}"
        
        # FIX: Load ONNX model with explicit providers
        try:
            # Try CPU provider first (most compatible)
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            return session, class_names, "✅ Model loaded successfully with CPU!"
        except Exception as e:
            st.error(f"CPU provider failed: {e}")
            # Fallback to any available provider
            session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
            return session, class_names, "✅ Model loaded with available providers!"
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def preprocess_image(image):
    """Preprocess image without torchvision"""
    try:
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA to RGB
            img_array = img_array[:, :, :3]
        
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
        st.error(f"❌ Preprocessing error: {e}")
        return None

def demo_classification(image, class_names):
    """Provide demo classification when model isn't available"""
    # Create realistic-looking probabilities
    base_probs = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25])
    
    # Add some randomness but keep it realistic
    noise = np.random.normal(0, 0.05, 6)
    probabilities = np.clip(base_probs + noise, 0.01, 0.99)
    probabilities = probabilities / np.sum(probabilities)  # Re-normalize
    
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    return predicted_class, confidence, probabilities

def main():
    # Load model
    session, class_names, status = load_model()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Model Status")
    st.sidebar.write(status)
    
    use_demo_mode = session is None
    
    if use_demo_mode:
        st.warning("🔧 **Running in Demo Mode** - Using sample classification data")
        st.info("""
        **To enable real classification, make sure your GitHub has:**
        
        📁 models/
          ├── ✅ model_info.json
          ├── ✅ class_names.json  
          └── ✅ material_classifier_efficientnet_b0_streamlit.onnx
        """)
    else:
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
        with st.spinner("🔍 Analyzing image..."):
            start_time = time.time()
            
            if use_demo_mode:
                # Demo classification
                predicted_class, confidence, probabilities = demo_classification(image, class_names)
                inference_time = 18.5  # Simulated realistic time
            else:
                # Real classification
                image_array = preprocess_image(image)
                
                if image_array is not None:
                    try:
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
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        # Fallback to demo mode
                        predicted_class, confidence, probabilities = demo_classification(image, class_names)
                        inference_time = 20.0
                        use_demo_mode = True
                else:
                    st.error("❌ Failed to process image")
                    return
            
            # Display results
            st.subheader("🎯 Classification Results")
            
            # Confidence color
            if confidence > 0.8:
                color = "#00cc00"
                emoji = "🎯"
                conf_text = "Very High Confidence"
            elif confidence > 0.6:
                color = "#ff9900"
                emoji = "✅"
                conf_text = "High Confidence"
            else:
                color = "#ff3333"
                emoji = "⚠️"
                conf_text = "Moderate Confidence"
            
            st.markdown(f"""
            <div style='background: #f0f2f6; padding: 25px; border-radius: 10px; text-align: center; border-left: 5px solid {color}; margin: 20px 0;'>
                <h2 style='color: {color}; margin: 0 0 10px 0;'>{emoji} {predicted_class.upper()}</h2>
                <p style='font-size: 1.5rem; margin: 10px 0; color: {color};'>{confidence:.1%} Confidence</p>
                <p style='color: #666; font-size: 1rem;'>{conf_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if use_demo_mode:
                st.info("💡 This is a demo prediction. Upload model files for real classification.")
            
            # Performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚡ Inference Time", f"{inference_time:.1f} ms")
            with col2:
                st.metric("📊 Throughput", f"{1000/inference_time:.1f} FPS")
            
            # Detailed analysis
            st.subheader("📈 Detailed Analysis")
            
            # Get top predictions
            probs_array = probabilities[0] if not use_demo_mode else probabilities
            top_indices = np.argsort(probs_array)[-3:][::-1]
            
            st.write("**Top Predictions:**")
            for idx in top_indices:
                prob = probs_array[idx] if not use_demo_mode else probabilities[idx]
                class_name = class_names[idx]
                
                # Create progress bar with label
                col_label, col_bar, col_value = st.columns([2, 5, 1])
                with col_label:
                    st.write(f"**{class_name.title()}**")
                with col_bar:
                    st.progress(float(prob))
                with col_value:
                    st.write(f"{prob:.1%}")
            
            # Show all probabilities in expander
            with st.expander("📋 View All Probabilities"):
                for i, class_name in enumerate(class_names):
                    prob = probs_array[i] if not use_demo_mode else probabilities[i]
                    st.write(f"**{class_name.title()}**: {prob:.1%}")

if __name__ == "__main__":
    main()
