import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# 1. Page Configuration (වෙබ් පිටුවේ සැකසුම්)
st.set_page_config(
    page_title="AI Image Classifier", 
    page_icon="🖼️",
    layout="centered"
)

# App එකේ ප්‍රධාන මාතෘකාවන්
st.title("🖼️ AI-Based Image Classification Application")
st.write("Upload an image to predict its class using the trained MobileNetV2 model.")
st.markdown("---")

# 2. Load Model and Class Names (Model එක සහ Class Names ආරක්ෂිතව Load කිරීම)
@st.cache_resource
def load_assets():
    # Trained Keras model එක load කිරීම
    model = tf.keras.models.load_model('model.keras')
    
    # Class names JSON file එක කියවීම
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
        
    return model, class_names

# Files දෙක හරියට load වෙනවාදැයි බැලීම
try:
    model, class_names = load_assets()
    st.sidebar.success("✅ Model & Class Names loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading assets: {e}")
    st.error("Please ensure 'model.keras' and 'class_names.json' are in the same directory.")
    st.stop()

# 3. File Uploader Component (User ට image එකක් දාන්න දෙන තැන)
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Upload කරපු image එක open කරගෙන screen එකේ පෙන්වීම
    image = Image.open(uploaded_file)
    
    # Layout එක ලස්සන කරන්න columns 2ක් භාවිතා කරමු
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, caption='Your Input Image', use_container_width=True)
        
    with col2:
        st.subheader("📊 Prediction Analysis")
        with st.spinner("Classifying the image... Please wait..."):
            
            # 4. Image Preprocessing (MobileNetV2 වලට ගැලපෙන සේ රූපය සකස් කිරීම)
            # MobileNetV2 බලාපොරොත්තු වන්නේ 224x224 ප්‍රමාණයේ රූප වේ
            img_resized = image.resize((224, 224)) 
            img_array = np.array(img_resized)
            
            # රූපයේ Alpha channel (RGBA) එකක් තිබේ නම් එය ඉවත් කර RGB කර ගැනීම
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            
            # Batch size එකක් ලෙස දැක්වීමට dimension එකක් එකතු කිරීම (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Pixels වල අගයන් -1 සහ 1 අතරට පත් කිරීම (MobileNetV2 Preprocessing)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # 5. Model Prediction (අනාවැකිය ලබා ගැනීම)
            predictions = model.predict(img_preprocessed)
            
            # Logits වලින් සම්භාවිතාවන් (Probabilities) ලබා ගැනීමට Softmax භාවිතය
            score = tf.nn.softmax(predictions[0]) 
            
            # වැඩිම සම්භාවිතාව ඇති Index එක සහ නම ලබා ගැනීම
            predicted_class_idx = np.argmax(score)
            predicted_class_name = class_names[predicted_class_idx]
            
            # විශ්වාසනීයත්ව ප්‍රතිශතය (Confidence Score)
            confidence = np.max(score) * 100
            
            # 6. Display Results (ප්‍රතිඵල ලස්සනට තිරය මත පෙන්වීම)
            st.metric(label="Predicted Target Class", value=f"{predicted_class_name}")
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
            # Progress bar එකක් මගින් visual feedback එකක් දීම
            st.progress(int(confidence))

st.markdown("---")
st.caption("Developed for Artificial Intelligence (Practical) - INTE 31063")