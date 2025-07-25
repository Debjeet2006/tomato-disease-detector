import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration with custom styling
st.set_page_config(
    page_title="🍅 Tomato Disease AI Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .upload-container {
        border: 3px dashed #4ecdc4;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(78, 205, 196, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .upload-container:hover {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ecdc4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_class_data():
    try:
        with open("class_indices.json", "r") as f:
            class_indices = json.load(f)
        return list(class_indices.keys())
    except FileNotFoundError:
        return [
            "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", 
            "Septoria_leaf_spot", "Two-spotted_spider_mite", "Target_Spot",
            "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", 
            "healthy", "powdery_mildew"
        ]

@st.cache_resource
def load_disease_model():
    try:
        return load_model("tomato_disease_model_v2.h5")
    except:
        st.error("⚠️ Model file not found! Please ensure 'tomato_disease_model_v2.h5' is in the same directory.")
        return None

class_names = load_class_data()
model = load_disease_model()

st.markdown("""
<div class="header-container">
    <h1 style="font-size: 3.5rem; margin-bottom: 0;">🍅 Tomato Disease AI Detector</h1>
    <p style="font-size: 1.2rem; margin-top: 0;">Advanced Machine Learning for Plant Health Diagnosis</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 Model Information")

    if model:
        st.markdown(f"""
        <div class="info-card">
            <h4>🎯 Model Stats</h4>
            <p><strong>Classes:</strong> {len(class_names)}</p>
            <p><strong>Input Size:</strong> 128x128 pixels</p>
            <p><strong>Architecture:</strong> CNN with Transfer Learning</p>
            <p><strong>Accuracy:</strong> ~85%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🌱 Disease Categories")

    disease_info = {
        "🦠 Bacterial Diseases": ["Bacterial_spot"],
        "🍂 Fungal Diseases": ["Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "powdery_mildew"],
        "🕷️ Pest Damage": ["Two-spotted_spider_mite", "Target_Spot"],
        "🦠 Viral Diseases": ["Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus"],
        "✅ Healthy": ["healthy"]
    }

    for category, diseases in disease_info.items():
        with st.expander(category):
            for disease in diseases:
                if disease in class_names:
                    st.write(f"• {disease.replace('_', ' ').title()}")

    st.markdown("## 📚 How to Use")
    st.markdown("""
    1. **Upload** a clear image of a tomato leaf
    2. **Wait** for the AI to analyze the image
    3. **Review** the prediction and confidence score
    4. **Get** treatment recommendations
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="upload-container">
        <h3>📤 Upload Tomato Leaf Image</h3>
        <p>Use your phone camera or file browser to upload an image</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],
                                     help="Upload a clear tomato leaf image")

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_treatment_info(disease):
    treatments = {
        "Bacterial_spot": {
            "severity": "Medium",
            "treatment": "Apply copper-based fungicides, improve air circulation",
            "prevention": "Use drip irrigation, avoid overhead watering"
        },
        "Early_blight": {
            "severity": "High",
            "treatment": "Remove affected leaves, apply fungicide spray",
            "prevention": "Crop rotation, proper spacing between plants"
        },
        "Late_blight": {
            "severity": "Critical",
            "treatment": "Immediate fungicide application, remove infected plants",
            "prevention": "Avoid wet conditions, use resistant varieties"
        },
        "healthy": {
            "severity": "None",
            "treatment": "No treatment needed - plant is healthy!",
            "prevention": "Continue good care practices"
        }
    }
    return treatments.get(disease, {
        "severity": "Unknown",
        "treatment": "Consult with agricultural specialist",
        "prevention": "Monitor plant regularly"
    })

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Analyzing your image..."):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_idx]
        confidence = float(np.max(prediction) * 100)
        top3_idx = np.argsort(prediction[0])[-3:][::-1]
        top3_preds = [(class_names[i], prediction[0][i] * 100) for i in top3_idx]

    with col2:
        info = get_treatment_info(predicted_class)
        color = {
            "None": "#4CAF50",
            "Medium": "#FF9800",
            "High": "#FF5722",
            "Critical": "#D32F2F",
            "Unknown": "#9E9E9E"
        }.get(info['severity'], "#9E9E9E")

        st.markdown(f"""
        <div class="result-card">
            <h3>🎯 Diagnosis</h3>
            <h2 style="color: #FFD700;">{predicted_class.replace('_', ' ').title()}</h2>
            <h4>Confidence: {confidence:.1f}%</h4>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                <p><strong>Severity:</strong> <span style="color: {color};">⚫ {info['severity']}</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ]}))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 📊 Top 3 Predictions")
    df = pd.DataFrame({
        "Disease": [i[0].replace('_', ' ').title() for i in top3_preds],
        "Confidence": [i[1] for i in top3_preds]
    })
    fig_bar = px.bar(df, x="Confidence", y="Disease", orientation="h",
                     color="Confidence", color_continuous_scale="Viridis")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("## 💊 Recommendations")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
        <div class="info-card">
            <h4>🏥 Treatment</h4>
            <p>{info['treatment']}</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="info-card">
            <h4>🛡️ Prevention</h4>
            <p>{info['prevention']}</p>
        </div>
        """, unsafe_allow_html=True)

    if confidence < 70:
        st.warning("⚠️ Low confidence. Try a clearer photo or consult an expert.")
    elif confidence > 90:
        st.success("✅ High confidence! This result is very reliable.")
    else:
        st.info("ℹ️ Moderate confidence. Further inspection recommended.")
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h3>🚀 Get Started</h3>
        <p>Upload a tomato leaf image to begin AI diagnosis!</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by Deep Learning | 🌱 For Agricultural Excellence</p>
    <p><small>This tool is for educational use. Consult experts for critical issues.</small></p>
</div>
""", unsafe_allow_html=True)
