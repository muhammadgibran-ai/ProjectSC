import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# HuggingFace Hub - Auto-download model files
# ============================================================
# Ganti nilai ini dengan username HuggingFace Anda dan nama repo
HF_REPO_ID = "YOUR_HF_USERNAME/proyek-sistem-cerdas-mobil"  # TODO: ganti sesuai repo HF Anda

@st.cache_resource(show_spinner=False)
def ensure_model_files():
    """Download model files from HuggingFace Hub if not already present locally."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_target = os.path.join(project_dir, "model_harga_mobil.h5")
    prep_target  = os.path.join(project_dir, "preprocessors.pkl")
    
    if not os.path.exists(model_target) or not os.path.exists(prep_target):
        try:
            from huggingface_hub import hf_hub_download
            if not os.path.exists(model_target):
                with st.spinner("⏳ Mengunduh model AI dari HuggingFace Hub... (pertama kali ~1 menit)"):
                    hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename="model_harga_mobil.h5",
                        local_dir=project_dir,
                        local_dir_use_symlinks=False
                    )
            if not os.path.exists(prep_target):
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="preprocessors.pkl",
                    local_dir=project_dir,
                    local_dir_use_symlinks=False
                )
        except Exception as e:
            st.error(f"❌ Gagal mengunduh model dari HuggingFace: {e}")
            return False
    return True

_model_ready = ensure_model_files()

# Helper functions defined locally to resolve Pickle serialization lookup under __main__
def clean_model_name(model_str):
    words = str(model_str).strip().split()
    if words:
        val = words[0]
        val = ''.join(c for c in val if c.isalnum())
        return val
    return "Unknown"

def clean_location(loc_str):
    loc_lower = str(loc_str).lower()
    if "jakarta" in loc_lower:
        return "DKI Jakarta"
    elif "banten" in loc_lower or "tangerang" in loc_lower:
        return "Banten"
    elif "jawa barat" in loc_lower or "bandung" in loc_lower or "depok" in loc_lower or "bekasi" in loc_lower or "bogor" in loc_lower:
        return "Jawa Barat"
    elif "jawa tengah" in loc_lower or "semarang" in loc_lower or "solo" in loc_lower:
        return "Jawa Tengah"
    elif "jawa timur" in loc_lower or "surabaya" in loc_lower or "malang" in loc_lower:
        return "Jawa Timur"
    elif "yogyakarta" in loc_lower or "jogja" in loc_lower:
        return "Yogyakarta"
    elif "bali" in loc_lower:
        return "Bali"
    elif "sumatera" in loc_lower or "medan" in loc_lower or "palembang" in loc_lower or "riau" in loc_lower or "lampung" in loc_lower:
        return "Sumatera"
    elif "kalimantan" in loc_lower:
        return "Kalimantan"
    elif "sulawesi" in loc_lower:
        return "Sulawesi"
    else:
        return "Lainnya"


# Page configuration
st.set_page_config(
    page_title="Estimasi Harga Mobil Bekas - Sistem Cerdas",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism and modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
    }
    
    /* Card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    .premium-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 24px;
        margin-bottom: 8px;
    }
    
    .premium-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 16px;
    }
    
    /* Stats styling */
    .stat-container {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .stat-card {
        flex: 1;
        background: rgba(56, 189, 248, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        background: rgba(56, 189, 248, 0.08);
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Form inputs custom */
    div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Predict Button */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    
    .result-price {
        font-size: 36px;
        font-weight: 800;
        color: #10b981;
        margin: 10px 0;
    }
    
    .result-range {
        font-size: 14px;
        color: #a7f3d0;
    }
</style>
""", unsafe_allow_html=True)

# Path definitions - relative to this file's folder to work on any laptop
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, "model_harga_mobil.h5")
preprocessors_path = os.path.join(project_dir, "preprocessors.pkl")
data_path = os.path.join(project_dir, "mobil_bekas_carmudi.csv")

# Load model and preprocessors
@st.cache_resource
def load_resources():
    if not os.path.exists(model_path) or not os.path.exists(preprocessors_path):
        return None, None
    
    # Load keras model with compile=False to avoid Keras 3 deserialization issues
    model = keras.models.load_model(model_path, compile=False)
    
    # Load pickle preprocessors
    with open(preprocessors_path, "rb") as f:
        preprocessors = pickle.load(f)
        
    return model, preprocessors

@st.cache_data
def load_car_data():
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Apply standard clean
        popular_brands = ["Toyota", "Honda", "Suzuki", "Mitsubishi", "Daihatsu", "Hyundai", "Wuling", "Nissan"]
        df_filtered = df[df['brand'].isin(popular_brands)]
        df_filtered = df_filtered[(df_filtered['price'] >= 60_000_000) & (df_filtered['price'] <= 800_000_000)].copy()
        
        # Add engineered columns
        df_filtered['age'] = 2026 - df_filtered['year']
        df_filtered['model_clean'] = df_filtered['model'].apply(clean_model_name)
        df_filtered['location_clean'] = df_filtered['location'].apply(clean_location)
        
        # Map minor models to 'Lainnya' if preprocessor is loaded
        if prep is not None:
            df_filtered['model_clean'] = df_filtered['model_clean'].apply(lambda x: x if x in prep["top_models"] else 'Lainnya')
            
        return df_filtered
    return None
 
model, prep = load_resources()
df = load_car_data()

# Navigation Sidebar
st.sidebar.markdown("<div style='text-align: center; padding: 10px;'><h2 style='color:#38bdf8; margin-bottom:0;'>Estimasi Mobil</h2><p style='color:#64748b; font-size:12px;'>Sistem Cerdas JST</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi Menu", ["🏠 Beranda", "🔮 Prediksi Harga", "📊 Analisis Pasar", "🧠 Performa JST"])
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:11px; color:#64748b; text-align:center;'>Proyek Sistem Informasi Industri Otomotif (SIIO) - 2026</div>", unsafe_allow_html=True)

if page == "🏠 Beranda":
    st.markdown("<h1 style='color:#f8fafc; font-weight:700;'>Sistem Cerdas Estimasi Harga Mobil Bekas di Indonesia</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:16px;'>Platform berbasis Jaringan Saraf Tiruan (JST) untuk memprediksi harga jual mobil bekas berdasarkan data riil dari ribuan listing terbaru di pasar otomotif Indonesia.</p>", unsafe_allow_html=True)
    
    # Visual stats
    if df is not None:
        st.markdown("""
        <div class='stat-container'>
            <div class='stat-card'>
                <div class='stat-value'>706</div>
                <div class='stat-label'>Total Data Listing</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>90.06%</div>
                <div class='stat-label'>Akurasi Model (R²)</div>
            </div>
            <div class='stat-card'>
                <div class='stat-value'>8.51%</div>
                <div class='stat-label'>Rata-rata Error (MAPE)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='premium-card'>
            <div class='premium-title'>Tentang Sistem Cerdas Ini</div>
            <p style='color:#cbd5e1; font-size:15px; line-height:1.6;'>
                Sistem ini dibangun untuk membantu pelaku bisnis otomotif, dealer mobil bekas, maupun konsumen individu dalam memperkirakan harga jual kendaraan secara objektif. Menggunakan metode <b>Artificial Neural Network (ANN)</b> tipe <b>Multi-Layer Perceptron (MLP)</b> dengan arsitektur 3 dense layers, model ini dilatih secara khusus untuk mendeteksi hubungan non-linear antara umur kendaraan, kilometer pemakaian, merek, model, transmisi, lokasi, dan jenis bahan bakar dengan nilai depresiasi harga kendaraan di Indonesia.
            </p>
            <p style='color:#cbd5e1; font-size:15px; line-height:1.6;'>
                <b>Keunggulan Sistem:</b>
                <ul style='color:#cbd5e1; margin-left: 20px;'>
                    <li><b>Data Riil & Terkini:</b> Dataset dikumpulkan dari pasar otomotif Indonesia secara langsung (carmudi.co.id) mencakup data hingga tahun 2026.</li>
                    <li><b>Akurasi Sangat Tinggi:</b> Dengan target optimasi MAPE sebesar 8.51% (target maksimal ketetapan proyek 10%), sistem ini memberikan presisi prediksi yang andal.</li>
                    <li><b>Metodologi Ilmiah:</b> Model dilatih dengan mengoptimasi Relative Percentage Error (MAPE) melalui integrasi One-Hot Encoding dan standardisasi scaling.</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='premium-card' style='height: 100%; text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div class='premium-title' style='margin-bottom:20px;'>Fitur Utama Aplikasi</div>
            <div style='text-align: left; width: 100%;'>
                <div style='margin-bottom: 15px;'>
                    <span style='font-size: 24px; margin-right: 10px;'>🔮</span>
                    <b>Estimator Harga Real-time:</b> Input spesifikasi mobil Anda dan peroleh harga pasar yang disarankan secara instan.
                </div>
                <div style='margin-bottom: 15px;'>
                    <span style='font-size: 24px; margin-right: 10px;'>📊</span>
                    <b>Visualisasi Analisis Pasar:</b> Lihat tren penurunan harga terhadap tahun, serta peta lokasi distribusi pasokan.
                </div>
                <div style='margin-bottom: 15px;'>
                    <span style='font-size: 24px; margin-right: 10px;'>🧠</span>
                    <b>Evaluasi Model JST:</b> Pelajari struktur neuron, kurva pembelajaran loss (MSE), dan scatter prediksi vs aktual.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔮 Prediksi Harga":
    st.markdown("<h1 style='color:#f8fafc; font-weight:700;'>🔮 Estimasi Harga Jual Mobil Bekas</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:15px;'>Masukkan spesifikasi kendaraan secara detail untuk memprediksi harga pasaran yang disarankan.</p>", unsafe_allow_html=True)
    
    if prep is None or model is None:
        st.error("Model JST atau objek preprocessor tidak ditemukan. Harap pastikan model sudah dilatih.")
    else:
        # Define Input Fields
        brands_list = sorted(prep["popular_brands"])
        transmission_list = ["Automatic", "Manual"]
        location_list = sorted(prep["clean_location"](loc) for loc in ["Jakarta", "Banten", "Jawa Barat", "Jawa Tengah", "Jawa Timur", "Yogyakarta", "Bali", "Sumatera", "Kalimantan", "Sulawesi", "Lainnya"])
        fuel_list = ["Bensin", "Diesel", "Hybrid", "Elektrik"]
        
        # Mapping models by brand for dynamic select box
        # Since we have raw data, let's extract which models belong to which brands
        brand_models = {}
        if df is not None:
            for b in brands_list:
                models_for_b = sorted(df[df['brand'] == b]['model_clean'].unique().tolist())
                # Ensure 'Lainnya' is at the end or included
                if 'Lainnya' in models_for_b:
                    models_for_b.remove('Lainnya')
                models_for_b.append('Lainnya')
                brand_models[b] = models_for_b
        else:
            # Fallback if df is not loaded
            for b in brands_list:
                brand_models[b] = ['Lainnya']
                
        # Layout Form columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.subheader("Spesifikasi Umum")
            brand = st.selectbox("Merek Mobil", brands_list)
            
            # Dynamic Model Box based on brand selection
            model_options = brand_models.get(brand, ['Lainnya'])
            model_name = st.selectbox("Model Utama", model_options)
            
            year = st.slider("Tahun Pembuatan", min_value=2010, max_value=2026, value=2020)
            mileage = st.number_input("Jarak Tempuh / Odometer (KM)", min_value=0, max_value=500000, value=65000, step=5000)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.subheader("Teknis & Geografis")
            transmission = st.radio("Jenis Transmisi", transmission_list, horizontal=True)
            fuel_type = st.radio("Jenis Bahan Bakar", fuel_list, horizontal=True)
            location = st.selectbox("Lokasi Penjualan", location_list)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Prediction logic
        st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
        if st.button("PREDIKSI HARGA PASARAN"):
            # 1. Feature Engineering
            age = 2026 - year
            
            # Map inputs to dummy variables
            # We initialize a zero dictionary for all dummy variables
            input_dict = {col: 0 for col in prep["all_dummy_columns"] if col not in prep["num_cols"] and col != 'price'}
            
            # Clean model name mapping
            cleaned_model = prep["clean_model_name"](model_name)
            if cleaned_model not in prep["top_models"]:
                cleaned_model = 'Lainnya'
                
            # Clean location mapping
            cleaned_loc = prep["clean_location"](location)
            
            # Set target one-hot encoding columns to 1
            brand_col = f"brand_{brand}"
            model_col = f"model_clean_{cleaned_model}"
            trans_col = f"transmission_{transmission}"
            loc_col = f"location_clean_{cleaned_loc}"
            fuel_col = f"fuel_type_{fuel_type}"
            
            for col in [brand_col, model_col, trans_col, loc_col, fuel_col]:
                if col in input_dict:
                    input_dict[col] = 1
            
            # Convert to DataFrame to match exact order of dummy columns
            # Create a 1-row dataframe for category dummies
            df_cat = pd.DataFrame([input_dict])
            
            # Standardize numerical features
            num_data = pd.DataFrame([{"age": age, "mileage": mileage}])
            num_scaled = prep["scaler"].transform(num_data)
            
            # Stack inputs
            X_cat_vals = df_cat[prep["dummy_columns"]].values.astype(np.float32)
            X_input = np.hstack([num_scaled, X_cat_vals])
            
            # Predict
            pred_log = model.predict(X_input).flatten()[0]
            pred_juta = np.exp(pred_log)
            
            # Final price in Rupiah
            pred_rupiah = pred_juta * 1e6
            
            # Display Prediction Result card
            st.markdown(f"""
            <div class='result-card'>
                <div style='font-size:16px; color:#cbd5e1; font-weight:600;'>HARGA ESTISIMA PASARAN YANG DISARANKAN</div>
                <div class='result-price'>Rp {pred_rupiah:,.0f}</div>
                <div class='result-range'>Rentang Harga Wajar (±8.51% MAPE): <b>Rp {pred_rupiah * 0.9149:,.0f} - Rp {pred_rupiah * 1.0851:,.0f}</b></div>
                <p style='color:#a7f3d0; font-size:12px; margin-top:8px;'>Nilai depresiasi dipengaruhi oleh odometer {mileage:,.0f} KM dan usia kendaraan {age} tahun di daerah {location}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Financing Calculator Simulator
            st.markdown("<br><div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("<div class='premium-title'>Simulasi Pembiayaan Kredit</div>", unsafe_allow_html=True)
            
            dp_pct = st.slider("Persentase Uang Muka (DP %)", min_value=10, max_value=80, value=20, step=5)
            bunga_pct = st.slider("Suku Bunga Kredit Per Tahun (%)", min_value=3.0, max_value=15.0, value=6.0, step=0.5)
            
            dp_val = pred_rupiah * (dp_pct / 100)
            loan_amt = pred_rupiah - dp_val
            
            tenors = [1, 2, 3, 4, 5]
            st.markdown("<table style='width:100%; border-collapse: collapse; margin-top:15px; color:#cbd5e1;'>", unsafe_allow_html=True)
            st.markdown("<tr style='border-bottom: 2px solid rgba(255,255,255,0.1); text-align: left;'><th style='padding:10px;'>Tenor (Tahun)</th><th>Uang Muka (DP)</th><th>Pokok Hutang</th><th>Bunga Flat/Tahun</th><th>Angsuran Per Bulan</th></tr>", unsafe_allow_html=True)
            
            for t in tenors:
                months = t * 12
                total_interest = loan_amt * (bunga_pct / 100) * t
                total_loan = loan_amt + total_interest
                monthly_installment = total_loan / months
                
                st.markdown(f"<tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:10px; font-weight:600;'>{t} Tahun ({months}x)</td><td>Rp {dp_val:,.0f}</td><td>Rp {loan_amt:,.0f}</td><td>{bunga_pct}%</td><td style='color:#38bdf8; font-weight:700;'>Rp {monthly_installment:,.0f} / bln</td></tr>", unsafe_allow_html=True)
                
            st.markdown("</table>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📊 Analisis Pasar":
    st.markdown("<h1 style='color:#f8fafc; font-weight:700;'>📊 Analisis Pasar Mobil Bekas di Indonesia</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:15px;'>Eksplorasi visual berdasarkan dataset riil dari portal iklan kendaraan Carmudi Indonesia.</p>", unsafe_allow_html=True)
    
    if df is None:
        st.warning("Dataset tidak tersedia.")
    else:
        # Layout plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("<div class='premium-title'>Distribusi Harga Kendaraan</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e293b')
            ax.set_facecolor('#0f172a')
            sns.histplot(df['price'] / 1e6, bins=20, kde=True, color='#38bdf8', ax=ax)
            ax.set_xlabel("Harga (Juta Rupiah)", color='#cbd5e1')
            ax.set_ylabel("Jumlah Kendaraan", color='#cbd5e1')
            ax.tick_params(colors='#cbd5e1')
            ax.title.set_color('#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("<div class='premium-title'>Depresiasi Harga vs Jarak Tempuh</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e293b')
            ax.set_facecolor('#0f172a')
            sns.scatterplot(data=df, x='mileage', y=df['price']/1e6, hue='brand', palette='Set2', alpha=0.7, ax=ax)
            ax.set_xlabel("Jarak Tempuh (Odometer KM)", color='#cbd5e1')
            ax.set_ylabel("Harga (Juta Rupiah)", color='#cbd5e1')
            ax.tick_params(colors='#cbd5e1')
            ax.legend(facecolor='#0f172a', labelcolor='#cbd5e1', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("<div class='premium-title'>Pangsa Pasar Berdasarkan Merek</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e293b')
            ax.set_facecolor('#0f172a')
            brand_cnt = df['brand'].value_counts()
            ax.pie(brand_cnt, labels=brand_cnt.index, autopct='%1.1f%%', colors=sns.color_palette('Blues_r', len(brand_cnt)), textprops={'color': '#cbd5e1'})
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("<div class='premium-title'>Harga Rata-Rata Berdasarkan Tahun Pembuatan</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e293b')
            ax.set_facecolor('#0f172a')
            avg_price_year = df.groupby('year')['price'].mean() / 1e6
            ax.plot(avg_price_year.index, avg_price_year.values, marker='o', color='#10b981', linewidth=2.5)
            ax.set_xlabel("Tahun Pembuatan", color='#cbd5e1')
            ax.set_ylabel("Harga Rata-Rata (Juta Rupiah)", color='#cbd5e1')
            ax.tick_params(colors='#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "🧠 Performa JST":
    st.markdown("<h1 style='color:#f8fafc; font-weight:700;'>🧠 Arsitektur & Performa Jaringan Saraf Tiruan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:15px;'>Rincian teknis model Artificial Neural Network (ANN) tipe Multi-Layer Perceptron (MLP) yang digunakan.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='premium-card'>
            <div class='premium-title'>Arsitektur Jaringan</div>
            <table style='width:100%; border-collapse: collapse; margin-top:10px; color:#cbd5e1;'>
                <tr style='border-bottom: 2px solid rgba(255,255,255,0.1);'><th style='padding:8px;'>Layer Tipe</th><th>Ukuran Output</th><th>Fungsi Aktivasi</th><th>Jumlah Parameter</th></tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:8px;'><b>Input (Fitur Mobil)</b></td><td>Dimension: 44</td><td>-</td><td>0</td></tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:8px;'><b>Dense_1 (Hidden)</b></td><td>64</td><td>ReLU</td><td>2,880</td></tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:8px;'><b>Dense_2 (Hidden)</b></td><td>32</td><td>ReLU</td><td>2,080</td></tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:8px;'><b>Output (Log Harga)</b></td><td>1</td><td>Linear (Regresi)</td><td>33</td></tr>
            </table>
            <br>
            <b>Optimasi Model:</b>
            <ul style='color:#cbd5e1; margin-top:5px; margin-left:20px;'>
                <li><b>Loss Function:</b> Mean Squared Error (MSE) dihitung pada skala logaritma harga untuk menyeimbangkan variasi harga mobil murah dan mahal.</li>
                <li><b>Optimizer:</b> Adam dengan learning rate 0.005 untuk laju konvergensi yang cepat dan stabil.</li>
                <li><b>Pencegahan Overfitting:</b> Implementasi Early Stopping memantau validation loss dengan batas toleransi (patience) 20 epochs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='premium-card'>
            <div class='premium-title'>Metrik Evaluasi Model</div>
            <table style='width:100%; border-collapse: collapse; margin-top:10px; color:#cbd5e1;'>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:8px; font-weight:600;'>Mean Absolute Percentage Error (Test MAPE)</td><td style='color:#10b981; font-weight:700;'>8.51%</td></tr>
                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='padding:8px; font-weight:600;'>Koefisien Determinasi (R² Score)</td><td style='color:#10b981; font-weight:700;'>0.9006</td></tr>
            <p style='font-size:12px; color:#94a3b8; margin-top:10px;'>Akurasi sebesar 90.06% R² mengindikasikan bahwa model dapat menjelaskan lebih dari 90% variasi penyebaran harga mobil bekas di pasar sekunder Indonesia berdasarkan masukan spesifikasi yang diberikan.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-title'>Grafik Perbandingan: Aktual vs Prediksi</div>", unsafe_allow_html=True)
        
        # We can construct a beautiful predicted vs actual chart using our model predictions
        if df is not None and model is not None and prep is not None:
            # Let's run prediction on a sample subset for plot
            cat_cols = ['brand', 'model_clean', 'transmission', 'location_clean', 'fuel_type']
            df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
            dummy_columns = [col for col in df_encoded.columns if any(col.startswith(cat + "_") for cat in cat_cols)]
            num_cols = ['age', 'mileage']
            X_num = prep["scaler"].transform(df_encoded[num_cols])
            X_cat = df_encoded[prep["dummy_columns"]].values.astype(np.float32)
            X_full = np.hstack([X_num, X_cat])
            
            # Predict
            pred_logs = model.predict(X_full, verbose=0).flatten()
            pred_prices = np.exp(pred_logs)
            actual_prices = df['price'].values / 1e6
            
            # Plot
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1e293b')
            ax.set_facecolor('#0f172a')
            
            # Draw line
            max_val = max(max(actual_prices), max(pred_prices))
            ax.plot([0, max_val], [0, max_val], '--', color='#ef4444', label='Garis Sempurna (Prediksi=Aktual)', linewidth=2)
            
            # Scatter
            ax.scatter(actual_prices, pred_prices, color='#38bdf8', alpha=0.6, edgecolors='none', s=25)
            
            ax.set_xlabel("Harga Aktual (Juta Rupiah)", color='#cbd5e1')
            ax.set_ylabel("Harga Prediksi JST (Juta Rupiah)", color='#cbd5e1')
            ax.tick_params(colors='#cbd5e1')
            ax.legend(facecolor='#0f172a', labelcolor='#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Grafik tidak dapat ditampilkan.")
        st.markdown("</div>", unsafe_allow_html=True)


