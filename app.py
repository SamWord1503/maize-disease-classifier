import streamlit as st
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import plotly.graph_objects as go

st.set_page_config(
    page_title="Maize Disease Classifier",
    page_icon="🌽",
    layout="centered"
)

# ─── SIDEBAR ───
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Maize_tassel.jpg/320px-Maize_tassel.jpg", 
             width=200)
    st.title("🌽 Navigation")
    st.markdown("---")
    
    page = st.radio("Menu", [
        "🏠 Home",
        "🌿 Disease Guide",
        "📊 Model Info",
        "👨‍💻 Developer"
    ])
    
    st.markdown("---")
    st.caption("Built with TechCrush AI/ML Program 🇳🇬")

# ─── DISEASE INFO ───
disease_info = {
    "Blight": {
        "description": "Northern Leaf Blight causes long tan lesions on leaves.",
        "treatment": "Apply fungicides and remove infected leaves.",
        "severity": "🟡 Moderate",
        "color": "#FFA500"
    },
    "Common_Rust": {
        "description": "Common Rust appears as orange/brown pustules on leaves.",
        "treatment": "Apply appropriate fungicides early.",
        "severity": "🟠 High",
        "color": "#FF4500"
    },
    "Gray_Leaf_Spot": {
        "description": "Gray Leaf Spot causes rectangular gray lesions.",
        "treatment": "Use resistant varieties and apply fungicides.",
        "severity": "🟡 Moderate",
        "color": "#808080"
    },
    "Healthy": {
        "description": "Your maize plant looks healthy!",
        "treatment": "No treatment needed. Keep monitoring.",
        "severity": "🟢 None",
        "color": "#008000"
    }
}

# ─── LOAD MODEL ───
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        'maize_disease_model.keras',
        compile=False
    )
    return model

model = load_model()
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# ─── HOME PAGE ───
if page == "🏠 Home":
    st.title("🌽 Maize Disease Classifier")
    st.write("Upload a maize leaf image to detect disease instantly!")

    uploaded_file = st.file_uploader(
        "Choose a maize leaf image",
        type=['jpg', 'jpeg', 'png', 'JPG']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("🔍 Analyse Leaf"):
            with st.spinner("Analysing your maize leaf..."):
                img = image.resize((224, 224)).convert('RGB')
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)
                confidence_scores = predictions[0]
                predicted_class = np.argmax(confidence_scores)
                predicted_disease = class_names[predicted_class]
                confidence = confidence_scores[predicted_class] * 100

            st.markdown("---")
            st.subheader("📋 Diagnosis Results")

            if predicted_disease == "Healthy":
                st.success(f"✅ DIAGNOSIS: {predicted_disease}")
            else:
                st.error(f"⚠️ DIAGNOSIS: {predicted_disease}")

            st.metric("Confidence Level", f"{confidence:.1f}%")

            info = disease_info[predicted_disease]
            st.markdown("---")
            st.write(f"*Severity:* {info['severity']}")
            st.write(f"*Description:* {info['description']}")
            st.write(f"*Recommended Treatment:* {info['treatment']}")

            st.markdown("---")
            st.subheader("📊 Confidence Breakdown")

            fig = go.Figure(go.Bar(
                x=class_names,
                y=[score * 100 for score in confidence_scores],
                marker_color=[disease_info[d]['color'] for d in class_names],
                text=[f"{score*100:.1f}%" for score in confidence_scores],
                textposition='auto',
            ))
            fig.update_layout(
                title="Disease Probability Distribution",
                xaxis_title="Disease",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.caption("Built with TechCrush AI/ML Program 🇳🇬")

# ─── DISEASE GUIDE PAGE ───
elif page == "🌿 Disease Guide":
    st.title("🌿 Maize Disease Guide")
    st.write("Learn about the 4 maize conditions our model detects!")
    st.markdown("---")

    for disease, info in disease_info.items():
        with st.expander(f"🔍 {disease}"):
            st.write(f"*Severity:* {info['severity']}")
            st.write(f"*Description:* {info['description']}")
            st.write(f"*Treatment:* {info['treatment']}")

# ─── MODEL INFO PAGE ───
elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "95.34%")
    with col2:
        st.metric("Model Type", "CNN")
    with col3:
        st.metric("Classes", "4")

    st.markdown("---")
    st.subheader("🧠 How it works")
    st.write("""
    This model uses *Transfer Learning* with *MobileNetV2* — 
    a powerful architecture pre-trained on 1.4 million images.
    
    *Training Details:*
    - Dataset: 4,188 maize leaf images
    - Training phases: 3 phases of fine tuning
    - Total epochs: 40
    - Best validation accuracy: 95.34%
    """)

    st.markdown("---")
    st.subheader("📈 Performance by Disease")
    
    diseases = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    accuracies = [92, 98, 86, 100]
    colors = ['#FFA500', '#FF4500', '#808080', '#008000']

    fig = go.Figure(go.Bar(
        x=diseases,
        y=accuracies,
        marker_color=colors,
        text=[f"{a}%" for a in accuracies],
        textposition='auto',
    ))
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Accuracy (%)",
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── DEVELOPER PAGE ───
elif page == "👨‍💻 Developer":
    st.title("👨‍💻 Developer")
    st.markdown("---")

    st.subheader("SamWord 🇳🇬")
    st.write("""
    *AI/ML Engineer in Training*
    
    Currently studying Artificial Intelligence and Machine Learning 
    at *TechCrush* — a 3-month intensive scholarship program.
    
   *Skills:**
    - 🐍 Python Programming
    - 📊 Data Preprocessing
    - 🤖 Machine Learning
    - 🧠 Deep Learning
    - 👁️ Computer Vision
    - 🚀 Model Deployment
    
    *This Project:*
    Built a Maize Disease Classification model using Transfer Learning 
    with MobileNetV2, achieving 95.34% accuracy on 4,188 images!
    """)

    st.markdown("---")
    st.caption("Built with TechCrush AI/ML Program 🇳🇬 | Powered by TensorFlow & Streamlit")
