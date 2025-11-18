import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load('lightGBM_model.pkl')
    return model

@st.cache_resource
def load_features():
    features = joblib.load('feature_names.pkl')
    return features

try:
    model = load_model()
    feature_names = load_features()
    st.success(f"✅ Model loaded successfully! Features: {len(feature_names)}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title and description
st.title("💻 Laptop Price Predictor")
st.markdown("### Enter laptop specifications to predict the price")
st.divider()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 Basic Specifications")
    
    brand = st.selectbox(
        "Brand", 
        ["ASUS", "Lenovo", "HP", "Dell", "Acer", "MSI", "Other", "Samsung", "Apple"],
        help="Select the laptop brand"
    )
    
    processor_brand = st.radio(
        "Processor Brand", 
        ["Intel", "AMD", "Apple"], 
        horizontal=True
    )
    
    processor_tier = st.selectbox(
        "Processor Tier", 
        ["Low-End", "Mid-End", "High-End", "Apple M-series"],
        help="Performance tier of the processor"
    )
    
    processor_speed = st.slider(
        "Processor Speed (GHz)", 
        min_value=1.0, 
        max_value=5.5, 
        value=2.5, 
        step=0.5,
        help="Base clock speed of the processor"
    )
    
    st.markdown("**Memory Configuration**")
    ram_col1, ram_col2 = st.columns(2)
    with ram_col1:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
    with ram_col2:
        ram_type = st.selectbox("RAM Type", [3, 4, 5], format_func=lambda x: f"DDR{x}")
    
    ram_expandable = st.radio("RAM Expandable", ["Yes", "No"], horizontal=True)

with col2:
    st.subheader("💾 Storage & Display")
    
    st.markdown("**Storage**")
    storage_col1, storage_col2 = st.columns(2)
    with storage_col1:
        ssd = st.selectbox(
            "SSD (GB)", 
            [0, 128, 256, 512, 1024], 
            format_func=lambda x: "No SSD" if x == 0 else f"{x} GB",
            index=2  # Default to 256GB
        )
    with storage_col2:
        hdd = st.selectbox(
            "HDD (GB)", 
            [0, 500, 1024], 
            format_func=lambda x: "No HDD" if x == 0 else f"{x} GB"
        )
    if ssd == 0 and hdd == 0:
        st.warning("⚠️ Please select at least one storage option (SSD or HDD).")
    
    
    st.markdown("**Graphics**")
    gpu_brand = st.radio(
        "GPU Brand", 
        ["Intel", "NVIDIA", "AMD", "Apple"], 
        horizontal=True
    )
    
    gpu_tier = st.selectbox(
        "GPU Tier", 
        ["Entry-level", "Low-end", "Mid-end", "High-end"],
        help="Performance tier of the graphics card"
    )
    
    st.markdown("**Display**")
    display_col1, display_col2 = st.columns(2)
    with display_col1:
        display_type = st.radio("Display Type", ["LCD", "LED"], horizontal=True)
    with display_col2:
        display_tier = st.radio("Display Size", ["Small", "Medium", "Large"], horizontal=True)

st.divider()

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    
     with st.spinner("Calculating price..."):
        try:
            # Create input dataframe with EXACT column names from training
            input_data = pd.DataFrame({
                'Brand': [brand],
                'Processor_Brand': [processor_brand],
                'Processor_Tier': [processor_tier],
                'Processor_Speed(Ghz)': [processor_speed],  # ✅ Match training
                'RAM': [ram],
                'RAM_TYPE(DDR)': [ram_type],  # ✅ Match training
                'RAM_Expandable': [ram_expandable],
                'SSD(GB)': [ssd],  # ✅ Match training
                'HDD(GB)': [hdd],  # ✅ Match training
                'GPU_Brand': [gpu_brand],
                'GPU_Tier': [gpu_tier],
                'Display_type': [display_type],
                'Display_Tier': [display_tier]
            })
            
            # Debug: Show what we're sending to the model
            st.write("Input data columns:", input_data.columns.tolist())
            
            # Apply the EXACT same encoding as training
            # 1. RAM_Expandable encoding
            input_data['RAM_Expandable'] = input_data['RAM_Expandable'].map({'Yes': 1, 'No': 0})
            
            # 2. Display_type encoding
            input_data['Display_type'] = input_data['Display_type'].map({'LCD': 0, 'LED': 1})
            
            # 3. Processor_Tier encoding
            tier_map = {'Low-End': 0, 'Mid-End': 1, 'High-End': 2, 'Apple M-series': np.nan}
            input_data['Processor_Tier'] = input_data['Processor_Tier'].map(tier_map)
            
            # 4. Display_Tier encoding
            input_data['Display_Tier'] = input_data['Display_Tier'].map({
                'Small': 1, 'Medium': 2, 'Large': 3
            })
            
            # 5. GPU_Tier encoding
            input_data['GPU_Tier'] = input_data['GPU_Tier'].map({
                'Entry-level': 1, 'Low-end': 2, 'Mid-end': 3, 'High-end': 4
            })
            
            # Debug: Show values before one-hot encoding
            st.write("Before one-hot encoding:")
            st.write(input_data)
            
            # 6. One-hot encoding for categorical variables
            input_data = pd.get_dummies(
                input_data, 
                columns=['Brand', 'Processor_Brand', 'GPU_Brand'], 
                drop_first=True, 
                prefix=['Brand', 'Processor', 'GPU']
            )
            
            # Debug: Show what columns we have after one-hot encoding
            st.write("After one-hot encoding:", input_data.columns.tolist())
            st.write("Expected features:", feature_names)
            
            # 7. Add missing columns with 0 (for categories not selected by user)
            missing_cols = []
            for col in feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
                    missing_cols.append(col)
            
            if missing_cols:
                st.write("Added missing columns:", missing_cols)
            
            # 8. Reorder columns to match training data exactly
            input_data = input_data[feature_names]
            
            # Debug: Show final input shape and sample values
            st.write("Final input shape:", input_data.shape)
            st.write("Final input values:")
            st.write(input_data)
            
            # 9. Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result with styling
            st.toast("✅ Prediction complete!") 
            
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            with result_col2:
                st.markdown("""
                    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
                        <h2 style='color: white; margin: 0;'>Predicted Price</h2>
                        <h1 style='color: #fff; font-size: 48px; margin: 10px 0;'>${:,.2f}</h1>
                    </div>
                """.format(prediction), unsafe_allow_html=True)
            
            # Show input summary in expandable section
            with st.expander("📋 View Complete Specifications"):
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("**Hardware**")
                    st.write(f"🏷️ Brand: **{brand}**")
                    st.write(f"⚙️ Processor: **{processor_brand} {processor_tier}**")
                    st.write(f"🔥 Speed: **{processor_speed} GHz**")
                    st.write(f"🧠 RAM: **{ram} GB DDR{ram_type}**")
                    st.write(f"📈 RAM Expandable: **{ram_expandable}**")
                
                with summary_col2:
                    st.markdown("**Storage & Display**")
                    st.write(f"💽 SSD: **{ssd} GB**" if ssd > 0 else "💽 SSD: **None**")
                    st.write(f"💿 HDD: **{hdd} GB**" if hdd > 0 else "💿 HDD: **None**")
                    st.write(f"🎮 GPU: **{gpu_brand} {gpu_tier}**")
                    st.write(f"🖥️ Display: **{display_type} - {display_tier}**")
            
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                with st.expander("📊 Top 10 Most Important Features"):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.dataframe(feature_importance, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check that all inputs are valid and try again.")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write("Expected features:", len(feature_names))
                st.write("Input data shape:", input_data.shape if 'input_data' in locals() else "N/A")
                st.write("Error details:", str(e))

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 This prediction is based on historical laptop data and machine learning.</p>
        <p>Actual prices may vary based on market conditions, promotions, and availability.</p>
    </div>
""", unsafe_allow_html=True)