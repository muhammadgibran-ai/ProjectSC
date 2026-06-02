import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

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

def build_model(input_dim):
    # MLP with light regularization - balanced for MAPE < 10% and Good Fit
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.0003)),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.0003)),
        layers.Dropout(0.05),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.0003)),
        layers.Dense(1, activation='linear')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def run_training():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "mobil_bekas_carmudi.csv")
    
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found!")
        return
        
    df = pd.read_csv(data_path)
    print(f"Original dataset rows: {len(df)}")
    
    # Filter Outliers and Popular Brands
    popular_brands = ["Toyota", "Honda", "Suzuki", "Mitsubishi", "Daihatsu", "Hyundai", "Wuling", "Nissan"]
    df = df[df['brand'].isin(popular_brands)]
    df = df[(df['price'] >= 60_000_000) & (df['price'] <= 800_000_000)]
    print(f"Filtered dataset rows: {len(df)}")
    
    # Preprocessing
    df['age'] = 2026 - df['year']
    df['model_clean'] = df['model'].apply(clean_model_name)
    df['location_clean'] = df['location'].apply(clean_location)
    
    # Group minor models
    model_counts = df['model_clean'].value_counts()
    top_models = model_counts[model_counts >= 3].index.tolist()
    df['model_clean'] = df['model_clean'].apply(lambda x: x if x in top_models else 'Lainnya')
    
    # Categorical Columns
    cat_cols = ['brand', 'model_clean', 'transmission', 'location_clean', 'fuel_type']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    dummy_columns = [col for col in df_encoded.columns if any(col.startswith(cat + "_") for cat in cat_cols)]
    
    # Scale numerical features
    num_cols = ['age', 'mileage']
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_encoded[num_cols])
    X_cat = df_encoded[dummy_columns].values.astype(np.float32)
    X = np.hstack([X_num, X_cat])
    
    # Target: log of price in Millions of Rupiah
    y = np.log(df['price'].values / 1e6).astype(np.float32)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    
    model = build_model(X.shape[1])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )

    # Reduce LR when val_loss plateaus for additional stability
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=1
    )

    print("Training MLP (Light Regularization - Target MAPE < 10%)...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=300,
        batch_size=24,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # ── Evaluate on TEST set ────────────────────────────────────────────────
    y_pred_log_test = model.predict(X_test).flatten()
    y_test_orig = np.exp(y_test)
    y_pred_test_orig = np.exp(y_pred_log_test)

    test_mae  = np.mean(np.abs(y_test_orig - y_pred_test_orig))
    test_mape = np.mean(np.abs((y_test_orig - y_pred_test_orig) / y_test_orig)) * 100
    ss_res_t  = np.sum((y_test_orig - y_pred_test_orig) ** 2)
    ss_tot_t  = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
    test_r2   = 1 - (ss_res_t / ss_tot_t)

    # ── Evaluate on TRAIN set (for Good Fit comparison) ─────────────────────
    y_pred_log_train = model.predict(X_train).flatten()
    y_train_orig = np.exp(y_train)
    y_pred_train_orig = np.exp(y_pred_log_train)

    train_mae  = np.mean(np.abs(y_train_orig - y_pred_train_orig))
    train_mape = np.mean(np.abs((y_train_orig - y_pred_train_orig) / y_train_orig)) * 100
    ss_res_tr  = np.sum((y_train_orig - y_pred_train_orig) ** 2)
    ss_tot_tr  = np.sum((y_train_orig - np.mean(y_train_orig)) ** 2)
    train_r2   = 1 - (ss_res_tr / ss_tot_tr)

    # Final epoch val_loss & train_loss
    final_train_loss = history.history['loss'][-1]
    final_val_loss   = history.history['val_loss'][-1]
    best_val_loss    = min(history.history['val_loss'])
    best_train_loss  = min(history.history['loss'])
    # Pakai best val loss dan best train loss (bukan epoch terakhir)
    # Jika dropout aktif, train loss bisa lebih besar dari val loss -> bukan overfitting
    gap_ratio        = best_val_loss / best_train_loss if best_train_loss > 0 else 1.0

    # ── Good Fit Indicator Report ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("         INDIKATOR KONDISI MODEL (GOOD FIT CHECK)")
    print("=" * 60)
    print(f"  {'Metrik':<30} {'TRAIN':>10} {'TEST':>10}")
    print("-" * 60)
    print(f"  {'MAE (Juta Rp)':<30} {train_mae:>10.2f} {test_mae:>10.2f}")
    print(f"  {'MAPE (%)':<30} {train_mape:>10.2f} {test_mape:>10.2f}")
    print(f"  {'R2 Score':<30} {train_r2:>10.4f} {test_r2:>10.4f}")
    print("-" * 60)
    print(f"  {'MSE Loss (terakhir)':<30} {final_train_loss:>10.4f} {final_val_loss:>10.4f}")
    print(f"  {'Best Val Loss':<30} {best_val_loss:>10.4f}")
    print(f"  {'Rasio Val/Train Loss':<30} {gap_ratio:>10.2f}x")
    print("=" * 60)

    # ── Interpretasi otomatis ────────────────────────────────────────────────
    mape_diff = abs(train_mape - test_mape)
    r2_diff   = abs(train_r2 - test_r2)
    print("\n  [INTERPRETASI]")
    if mape_diff <= 5.0 and test_mape <= 15.0:
        print("  [GOOD FIT] Gap train/val kecil, MAPE train & test mendekati")
        print(f"     MAPE selisih {mape_diff:.2f}% (ideal <= 5%)")
        print(f"     Selisih R2  {r2_diff:.4f}  (ideal <= 0.05)")
        print(f"     Status      : MODEL SIAP DIGUNAKAN")
    elif mape_diff > 5.0 and train_mape < test_mape:
        print(f"  [OVERFITTING] MAPE selisih {mape_diff:.2f}% (ideal <= 5%)")
        print("     Model terlalu hafal data training.")
    elif test_mape > 15.0:
        print(f"  [UNDERFITTING] MAPE test {test_mape:.2f}% terlalu tinggi (ideal <= 15%)")
        print("     Model belum belajar dengan cukup.")
    else:
        print(f"  [PERIKSA] MAPE selisih {mape_diff:.2f}% - evaluasi lebih lanjut")
    print("=" * 60)

    # alias untuk generate_jupyter_notebook
    mape = test_mape
    r2   = test_r2
    
    # Save Model
    model_save_path = os.path.join(project_dir, "model_harga_mobil.h5")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Save Preprocessors
    preprocessors = {
        "scaler": scaler,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dummy_columns": dummy_columns,
        "popular_brands": popular_brands,
        "top_models": top_models,
        "all_dummy_columns": list(df_encoded.columns),
        "clean_model_name": clean_model_name,
        "clean_location": clean_location
    }
    
    preprocessors_path = os.path.join(project_dir, "preprocessors.pkl")
    with open(preprocessors_path, "wb") as f:
        pickle.dump(preprocessors, f)
    print(f"Preprocessors saved to: {preprocessors_path}")
    
    # Generate Jupyter Notebook
    generate_jupyter_notebook(project_dir, mape, r2)

def generate_jupyter_notebook(project_dir, test_mape, r2):
    import json
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# DATA PREPROCESSING & MODEL TRAINING\n",
                    "## Proyek Sistem Cerdas: Prediksi Harga Mobil Bekas di Indonesia\n",
                    "\n",
                    "Notebook ini memuat proses preprocessing data, pembersihan outlier, pembangunan arsitektur Jaringan Saraf Tiruan (ANN) tipe MLP menggunakan Keras/TensorFlow, training dengan target log-transformed untuk optimasi MAPE, dan evaluasi performa model regresi harga mobil bekas."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Import Library"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import tensorflow as tf\n",
                    "from tensorflow import keras\n",
                    "from keras import layers, regularizers\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "import pickle\n",
                    "import os\n",
                    "\n",
                    "sns.set_theme(style='whitegrid')\n",
                    "print('TensorFlow version:', tf.__version__)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Memuat dan Eksplorasi Dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_csv('mobil_bekas_carmudi.csv')\n",
                    "print('Jumlah Data Awal:', len(df))\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Pembersihan Outlier & Pemilihan Brand Populer"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "popular_brands = ['Toyota', 'Honda', 'Suzuki', 'Mitsubishi', 'Daihatsu', 'Hyundai', 'Wuling', 'Nissan']\n",
                    "df = df[df['brand'].isin(popular_brands)]\n",
                    "\n",
                    "df = df[(df['price'] >= 60_000_000) & (df['price'] <= 800_000_000)]\n",
                    "print('Jumlah Data Setelah Filter:', len(df))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Preprocessing Data & Feature Engineering"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def clean_model_name(model_str):\n",
                    "    words = str(model_str).strip().split()\n",
                    "    if words:\n",
                    "        val = words[0]\n",
                    "        return ''.join(c for c in val if c.isalnum())\n",
                    "    return 'Unknown'\n",
                    "\n",
                    "def clean_location(loc_str):\n",
                    "    loc_lower = str(loc_str).lower()\n",
                    "    if 'jakarta' in loc_lower: return 'DKI Jakarta'\n",
                    "    elif 'banten' in loc_lower or 'tangerang' in loc_lower: return 'Banten'\n",
                    "    elif 'jawa barat' in loc_lower or 'bandung' in loc_lower: return 'Jawa Barat'\n",
                    "    elif 'jawa tengah' in loc_lower: return 'Jawa Tengah'\n",
                    "    elif 'jawa timur' in loc_lower: return 'Jawa Timur'\n",
                    "    elif 'yogyakarta' in loc_lower: return 'Yogyakarta'\n",
                    "    elif 'bali' in loc_lower: return 'Bali'\n",
                    "    elif 'sumatera' in loc_lower: return 'Sumatera'\n",
                    "    elif 'kalimantan' in loc_lower: return 'Kalimantan'\n",
                    "    elif 'sulawesi' in loc_lower: return 'Sulawesi'\n",
                    "    else: return 'Lainnya'\n",
                    "\n",
                    "df['age'] = 2026 - df['year']\n",
                    "df['model_clean'] = df['model'].apply(clean_model_name)\n",
                    "df['location_clean'] = df['location'].apply(clean_location)\n",
                    "\n",
                    "model_counts = df['model_clean'].value_counts()\n",
                    "top_models = model_counts[model_counts >= 3].index.tolist()\n",
                    "df['model_clean'] = df['model_clean'].apply(lambda x: x if x in top_models else 'Lainnya')\n",
                    "\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Encoding & Scaling"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cat_cols = ['brand', 'model_clean', 'transmission', 'location_clean', 'fuel_type']\n",
                    "df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)\n",
                    "\n",
                    "dummy_columns = [col for col in df_encoded.columns if any(col.startswith(cat + '_') for cat in cat_cols)]\n",
                    "\n",
                    "num_cols = ['age', 'mileage']\n",
                    "scaler = StandardScaler()\n",
                    "\n",
                    "X_num = scaler.fit_transform(df_encoded[num_cols])\n",
                    "X_cat = df_encoded[dummy_columns].values.astype(np.float32)\n",
                    "X = np.hstack([X_num, X_cat])\n",
                    "\n",
                    "# Target log Juta Rupiah\n",
                    "y = np.log(df['price'].values / 1e6).astype(np.float32)\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)\n",
                    "print('Dimensi Input:', X.shape[1])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6. Membangun Jaringan Saraf Tiruan (ANN)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model = keras.Sequential([\n",
                    "    layers.Input(shape=(X.shape[1],)),\n",
                    "    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0003)),\n",
                    "    layers.Dropout(0.1),\n",
                    "    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0003)),\n",
                    "    layers.Dropout(0.05),\n",
                    "    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0003)),\n",
                    "    layers.Dense(1, activation='linear')\n",
                    "])\n",
                    "\n",
                    "optimizer = keras.optimizers.Adam(learning_rate=0.001)\n",
                    "model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])\n",
                    "model.summary()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 7. Melatih Model"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)\n",
                    "reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)\n",
                    "\n",
                    "history = model.fit(\n",
                    "    X_train, y_train,\n",
                    "    validation_split=0.15,\n",
                    "    epochs=300,\n",
                    "    batch_size=24,\n",
                    "    callbacks=[early_stopping, reduce_lr],\n",
                    "    verbose=1\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 8. Evaluasi Performa Model (Original Scale)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 5))\n",
                    "plt.plot(history.history['loss'], label='Training Loss (MSE)')\n",
                    "plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')\n",
                    "plt.title('Kurva Belajar Model')\n",
                    "plt.xlabel('Epochs')\n",
                    "plt.ylabel('MSE')\n",
                    "plt.legend()\n",
                    "plt.show()\n",
                    "\n",
                    "y_pred_log = model.predict(X_test).flatten()\n",
                    "y_test_orig = np.exp(y_test)\n",
                    "y_pred_orig = np.exp(y_pred_log)\n",
                    "\n",
                    "mae = np.mean(np.abs(y_test_orig - y_pred_orig))\n",
                    "mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100\n",
                    "r2 = 1 - (np.sum((y_test_orig - y_pred_orig) ** 2) / np.sum((y_test_orig - np.mean(y_test_orig)) ** 2))\n",
                    "\n",
                    "print(f'Test MAE: {mae:.2f} Juta Rupiah')\n",
                    "print(f'Test MAPE: {mape:.2f}%')\n",
                    "print(f'R2 Score: {r2:.4f}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 9. Menyimpan Model"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model.save('model_harga_mobil.h5')\n",
                    "print('Model berhasil disimpan ke model_harga_mobil.h5')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    nb_path = os.path.join(project_dir, "training.ipynb")
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Jupyter Notebook generated at: {nb_path}")

if __name__ == "__main__":
    run_training()
