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
            st.error(
                f"Model files for {model_name} not found. "
                f"Ensure {model_name}_model.pkl and features_{model_name}.pkl exist."
            )
    
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
    df = pd.DataFrame([input_data])
    
    df['RAM_Expandable'] = df['RAM_Expandable'].map({'Yes': 1, 'No': 0})
    df['Display_type'] = df['Display_type'].map({'LCD': 0, 'LED': 1})
    df['Display_Tier'] = df['Display_Tier'].map({'Small': 1, 'Large': 2})
    df['GPU_Tier'] = df['GPU_Tier'].map({
        'Entry-level': 1, 'Low-end': 2, 'Mid-end': 3, 'High-end': 4
    })
    df['RAM_TYPE(DDR)'] = df['RAM_TYPE(DDR)'].map({
        '3': 1, 'LP3': 2, '4': 3, 'LP4': 4, '5': 5, 'LP5': 6
    })
    
    df_encoded = pd.get_dummies(
        df,
        columns=['Processor_Category', 'Brand', 'Processor_Brand', 'GPU_Brand'],
        drop_first=True
    )
    
    return df_encoded


def prepare_input(df_encoded, feature_list):
    for feature in feature_list:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    return df_encoded[feature_list]

# ========================
# Sidebar - Model Selection
# ========================
st.sidebar.header("⚙️ Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Model:",
    options=['linear', 'polynomial', 'randomforest', 'xgboost', 'lightgbm'],
    index=0
)

# ========================
# Processor Category Mapping
# ========================
processor_categories = {
    'Intel': [
        'Intel I3', 'Intel I5', 'Intel I7', 'Intel I9',
        'Intel Ultra 5', 'Intel Ultra 7', 'Intel Ultra 9',
        'Intel Pentium', 'Intel Celeron', 'Intel Other'
    ],
    'AMD': [
        'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7',
        'AMD Ryzen 9', 'AMD Athlon', 'AMD A-Series', 'AMD Other'
    ],
    'Apple': ['Apple M-series']
}

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

with col1:
    st.subheader("Brand & Processor")
    brand = st.selectbox(
        "Brand",
        ['ASUS', 'Lenovo', 'HP', 'Dell', 'Acer', 'MSI', 'Other', 'Samsung', 'Apple']
    )
    
    processor_brand = st.selectbox(
        "Processor Brand",
        ['Intel', 'AMD', 'Apple']
    )
    
    processor_category = st.selectbox(
        "Processor Category",
        processor_categories[processor_brand]
    )

with col2:
    st.subheader("Memory & Display")
    ram_capacity = st.selectbox("RAM Capacity (GB)", [4, 8, 16, 32])
    ram_expandable = st.radio("RAM Expandable", ['Yes', 'No'])
    ram_type = st.selectbox("RAM Type (DDR)", ['3', '4', '5', 'LP3', 'LP4', 'LP5'])

with col3:
    st.subheader("Storage & Performance")
    processor_speed = st.slider(
        "Processor Speed (GHz)", 1.0, 5.5, 2.5, 0.5
    )
    
    ssd_display = st.selectbox("SSD Storage", list(ssd_display_map.keys()))
    ssd_storage = ssd_display_map[ssd_display]
    
    hdd_display = st.selectbox("HDD Storage", list(hdd_display_map.keys()))
    hdd_storage = hdd_display_map[hdd_display]

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Display")
    display_type = st.radio("Display Type", ['LCD', 'LED'])
    display_tier = st.selectbox("Display Tier", ['Small', 'Large'])

with col5:
    st.subheader("GPU")
    gpu_brand = st.selectbox("GPU Brand", ['Intel', 'NVIDIA', 'AMD', 'Apple'])
    gpu_tier = st.selectbox(
        "GPU Tier", ['Low-end', 'Entry-level', 'Mid-end', 'High-end']
    )

# ========================
# Prediction Section
# ========================
if st.button("🚀 Predict Price", use_container_width=True, type="primary"):
    
    # ❗ VALIDATION: Prevent No SSD + No HDD
    if ssd_storage == 0 and hdd_storage == 0:
        st.error("❌ Invalid input: A laptop must have at least SSD or HDD storage.")
        st.stop()
    
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
        df_encoded = encode_input(input_data)
        model = models[selected_model]
        feature_list = features_dict[selected_model]
        df_encoded = prepare_input(df_encoded, feature_list)
        
        prediction = model.predict(df_encoded)[0]
        
        st.success("✅ Prediction Complete!")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Price", f"${prediction:,.2f}")
        with c2:
            st.info(f"**Model Used:** {selected_model.upper()}")
        
        with st.expander("📋 Input Summary"):
            st.table(pd.DataFrame(input_data.items(), columns=["Feature", "Value"]))
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# ========================
# Footer
# ========================
st.divider()
st.markdown("""
### 📝 Notes:
- Scaling is handled inside saved pipelines
- Feature order is enforced automatically
- Invalid storage combinations are blocked
""")
