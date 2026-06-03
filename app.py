import streamlit as st
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

st.set_page_config(
    page_title="Maize Disease Classifier",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Maize Disease Classifier")
st.write("Upload a maize leaf image to detect disease")

disease_info = {
    "Blight": {
        "description": "Northern Leaf Blight causes long tan lesions on leaves.",
        "treatment": "Apply fungicides and remove infected leaves.",
        "severity": "🟡 Moderate"
    },
    "Common_Rust": {
        "description": "Common Rust appears as orange/brown pustules on leaves.",
        "treatment": "Apply appropriate fungicides early.",
        "severity": "🟠 High"
    },
    "Gray_Leaf_Spot": {
        "description": "Gray Leaf Spot causes rectangular gray lesions.",
        "treatment": "Use resistant varieties and apply fungicides.",
        "severity": "🟡 Moderate"
    },
    "Healthy": {
        "description": "Your maize plant looks healthy!",
        "treatment": "No treatment needed. Keep monitoring.",
        "severity": "🟢 None"
    }
}

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('maize_disease_model.keras')
    return model

model = load_model()
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

uploaded_file = st.file_uploader(
    "Choose a maize leaf image",
    type=['jpg', 'jpeg', 'png', 'JPG']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Analyse Leaf"):
        with st.spinner("Analysing..."):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
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
        for i, disease in enumerate(class_names):
            st.progress(
                float(confidence_scores[i]),
                text=f"{disease}: {confidence_scores[i]*100:.1f}%"
            )

        st.markdown("---")
        st.caption("Built with TechCrush AI/ML Program 🇳🇬")
