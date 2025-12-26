import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💻 Laptop Price Predictor")
st.markdown("Predict laptop prices using machine learning models")


@st.cache_resource
def load_models_and_features():
    """Load all models and their corresponding feature lists"""
    # models_dir = Path("models")
    BASE_DIR = Path(__file__).resolve().parent.parent
    models_dir = BASE_DIR / "models"
    
    models = {}
    features = {}
    
    model_names = ['linear', 'polynomial', 'randomforest', 'xgboost', 'lightgbm']
    
    for model_name in model_names:
        try:
            model_path = models_dir / f"{model_name}_model.pkl"
            features_path = models_dir / f"features_{model_name}.pkl"
            
            models[model_name] = joblib.load(model_path)
            features[model_name] = joblib.load(features_path)
        except FileNotFoundError:
            st.error(f"Model files for {model_name} not found. Please ensure {model_name}_model.pkl and features_{model_name}.pkl exist in the 'models' directory.")
    
    return models, features

# Load models
try:
    models, features_dict = load_models_and_features()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ========================
# Encoding Functions
# ========================
def encode_input(input_data):
    """Encode categorical features same as training data"""
    df = pd.DataFrame([input_data])
    
    # Ordinal encoding for specific features
    df['RAM_Expandable'] = df['RAM_Expandable'].map({'Yes': 1, 'No': 0})
    df['Display_type'] = df['Display_type'].map({'LCD': 0, 'LED': 1})
    df['Display_Tier'] = df['Display_Tier'].map({'Small': 1, 'Large': 2})
    df['GPU_Tier'] = df['GPU_Tier'].map({
        'Entry-level': 1, 'Low-end': 2, 'Mid-end': 3, 'High-end': 4
    })
    df['RAM_TYPE(DDR)'] = df['RAM_TYPE(DDR)'].map({
        '3': 1, 'LP3': 2, '4': 3, 'LP4': 4, '5': 5, 'LP5': 6
    })
    
    # One-hot encoding
    df_encoded = pd.get_dummies(
        df,
        columns=['Processor_Category', 'Brand', 'Processor_Brand', 'GPU_Brand'],
        drop_first=True
    )
    
    return df_encoded

def prepare_input(df_encoded, feature_list):
    """Ensure the encoded dataframe has all required features in correct order"""
    # Add missing columns with 0 values
    for feature in feature_list:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Select only required features in correct order
    df_encoded = df_encoded[feature_list]
    
    return df_encoded

# ========================
# Sidebar - Model Selection
# ========================
st.sidebar.header("⚙️ Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Model:",
    options=['linear', 'polynomial', 'randomforest', 'xgboost', 'lightgbm'],
    index=0,  # linear as default
    help="Select the machine learning model for prediction"
)

# ========================
# Processor Category Mapping
# ========================
processor_categories = {
    'Intel': ['Intel I3', 'Intel I5', 'Intel I7', 'Intel I9', 
              'Intel Ultra 5', 'Intel Ultra 7', 'Intel Ultra 9',
              'Intel Pentium', 'Intel Celeron', 'Intel Other'],
    'AMD': ['AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9',
            'AMD Athlon', 'AMD A-Series', 'AMD Other'],
    'Apple': ['Apple M-series']
}

# Storage Display Mapping
ssd_display_map = {
    'No SSD': 0,
    '128 GB': 128,
    '256 GB': 256,
    '512 GB': 512,
    '1024 GB (1 TB)': 1024
}

hdd_display_map = {
    'No HDD': 0,
    '500 GB': 500,
    '1024 GB (1 TB)': 1024
}

# ========================
# Main Input Section
# ========================
st.header(f"📊 Input Features ({selected_model.upper()} Model)")

col1, col2, col3 = st.columns(3)

# Categorical Features
with col1:
    st.subheader("Brand & Processor")
    brand = st.selectbox(
        "Brand",
        options=['ASUS', 'Lenovo', 'HP', 'Dell', 'Acer', 'MSI', 'Other', 'Samsung', 'Apple']
    )
    
    processor_brand = st.selectbox(
        "Processor Brand",
        options=['Intel', 'AMD', 'Apple']
    )
    
    # Conditional processor category based on processor brand
    available_categories = processor_categories[processor_brand]
    processor_category = st.selectbox(
        "Processor Category",
        options=available_categories,
        help=f"Showing categories for {processor_brand} processors"
    )

with col2:
    st.subheader("Memory & Display")
    ram_capacity = st.selectbox(
        "RAM Capacity (GB)",
        options=[4, 8, 16, 32]
    )
    
    ram_expandable = st.radio(
        "RAM Expandable",
        options=['Yes', 'No']
    )
    
    ram_type = st.selectbox(
        "RAM Type (DDR)",
        options=['3', '4', '5', 'LP3', 'LP4', 'LP5']
    )

with col3:
    st.subheader("Storage & Performance")
    processor_speed = st.slider(
        "Processor Speed (GHz)",
        min_value=1.0,
        max_value=5.5,
        step=0.5,
        value=2.5,
        help="Drag to select processor speed"
    )
    
    ssd_display = st.selectbox(
        "SSD Storage",
        options=list(ssd_display_map.keys())
    )
    ssd_storage = ssd_display_map[ssd_display]
    
    hdd_display = st.selectbox(
        "HDD Storage",
        options=list(hdd_display_map.keys())
    )
    hdd_storage = hdd_display_map[hdd_display]

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Display")
    display_type = st.radio(
        "Display Type",
        options=['LCD', 'LED']
    )
    
    display_tier = st.selectbox(
        "Display Tier",
        options=['Small', 'Large']
    )

with col5:
    st.subheader("GPU")
    gpu_brand = st.selectbox(
        "GPU Brand",
        options=['Intel', 'NVIDIA', 'AMD', 'Apple']
    )
    
    gpu_tier = st.selectbox(
        "GPU Tier",
        options=['Low-end', 'Entry-level', 'Mid-end', 'High-end']
    )

# ========================
# Prediction Section
# ========================
if st.button("🚀 Predict Price", use_container_width=True, type="primary"):
    # Prepare input data
    input_data = {
        'Brand': brand,
        'Processor_Brand': processor_brand,
        'Processor_Category': processor_category,
        'RAM_Capacity': ram_capacity,
        'RAM_Expandable': ram_expandable,
        'RAM_TYPE(DDR)': ram_type,
        'Processor_Speed(Ghz)': processor_speed,
        'SSD(GB)': ssd_storage,
        'HDD(GB)': hdd_storage,
        'Display_type': display_type,
        'Display_Tier': display_tier,
        'GPU_Brand': gpu_brand,
        'GPU_Tier': gpu_tier
    }
    
    try:
        # Encode input
        df_encoded = encode_input(input_data)
        
        # Get model and its features
        model = models[selected_model]
        feature_list = features_dict[selected_model]
        
        # Prepare input with correct feature order
        df_encoded = prepare_input(df_encoded, feature_list)
        
        # Make prediction
        prediction = model.predict(df_encoded)[0]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Show predicted price with nice formatting
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Price",
                value=f"${prediction:,.2f}",
                delta=None
            )
        
        with col2:
            st.info(f"**Model Used:** {selected_model.upper()}")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            summary_df = pd.DataFrame(list(input_data.items()), columns=['Feature', 'Value'])
            st.table(summary_df)
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.error("Please check your model files and feature lists are correctly saved.")

# ========================
# Footer
# ========================
st.divider()
st.markdown("""
### 📝 Notes:
- **Linear & Polynomial Models:** Scaling is handled by the pipeline embedded in the model
- **Feature Order:** Automatically matched to each model's training features
- **Categorical Features:** Automatically one-hot encoded to match training format
- **Processor Category:** Dynamically filtered based on selected Processor Brand
- **Storage Options:** Display user-friendly labels (No SSD, 128 GB, etc.) while internally mapping to numeric values
""")