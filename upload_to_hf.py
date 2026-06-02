"""
Script untuk upload model ke HuggingFace Hub.

Cara pakai:
1. Install huggingface_hub: pip install huggingface_hub
2. Login ke HF: huggingface-cli login
3. Jalankan script ini: python upload_to_hf.py
"""

import os
import sys
from huggingface_hub import HfApi, create_repo

# ============================================================
# KONFIGURASI - Ganti sesuai akun HuggingFace Anda
# ============================================================
HF_USERNAME = input("Masukkan username HuggingFace Anda: ").strip()
REPO_NAME = "proyek-sistem-cerdas-mobil"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# File yang akan diupload
project_dir = os.path.dirname(os.path.abspath(__file__))
FILES_TO_UPLOAD = [
    "model_harga_mobil.h5",
    "preprocessors.pkl",
]

def main():
    api = HfApi()
    
    print(f"\n📦 Membuat repository di HuggingFace: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository berhasil dibuat/ditemukan: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Gagal membuat repository: {e}")
        sys.exit(1)
    
    for filename in FILES_TO_UPLOAD:
        filepath = os.path.join(project_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File tidak ditemukan, skip: {filepath}")
            continue
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\n⬆️  Mengupload: {filename} ({file_size_mb:.1f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"✅ {filename} berhasil diupload!")
        except Exception as e:
            print(f"❌ Gagal upload {filename}: {e}")
    
    print(f"\n🎉 Selesai! Model tersedia di: https://huggingface.co/{REPO_ID}")
    print(f"\n📝 PENTING: Update baris ini di app.py:")
    print(f'   HF_REPO_ID = "{REPO_ID}"')

if __name__ == "__main__":
    main()
