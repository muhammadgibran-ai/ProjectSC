"""
Script untuk membuat GitHub Release dan mengupload model sebagai release asset.

Cara pakai:
1. Buka https://github.com/settings/tokens/new
2. Buat token dengan scope: repo (full control)
3. Jalankan: python create_github_release.py
   Masukkan token ketika diminta.
"""

import os
import sys
import json
import requests

# ============================================================
GITHUB_REPO = "muhammadgibran-ai/ProjectSC"
RELEASE_TAG = "v1.0.0"
RELEASE_NAME = "Model Release v1.0 - Sistem Cerdas Estimasi Harga Mobil"
RELEASE_BODY = """
## Model Files Release

Release ini berisi file model terlatih yang digunakan oleh aplikasi Streamlit.

### Isi Release:
- `model_harga_mobil.h5` - Model ANN/MLP terlatih (TensorFlow/Keras)
- `preprocessors.pkl` - Objek preprocessing (StandardScaler, encoder, dll.)

### Performa Model:
- **R² Score**: 90.06%
- **MAPE**: 8.51%
- **Dataset**: 706 listing mobil bekas dari Carmudi Indonesia (2026)
"""

FILES_TO_UPLOAD = [
    "model_harga_mobil.h5",
    "preprocessors.pkl",
]
# ============================================================

def create_release(token: str) -> dict:
    """Buat release baru di GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "tag_name": RELEASE_TAG,
        "name": RELEASE_NAME,
        "body": RELEASE_BODY,
        "draft": False,
        "prerelease": False,
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    
    # Jika release sudah ada, ambil yang existing
    if resp.status_code == 422:
        print(f"⚠️  Release {RELEASE_TAG} sudah ada. Menggunakan yang sudah ada...")
        url_get = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}"
        resp = requests.get(url_get, headers=headers)
        resp.raise_for_status()
        return resp.json()
    
    resp.raise_for_status()
    return resp.json()


def upload_asset(token: str, upload_url: str, filepath: str, filename: str):
    """Upload file sebagai release asset."""
    # upload_url biasanya: https://uploads.github.com/repos/.../releases/.../assets{?name,label}
    upload_url_clean = upload_url.split("{")[0]
    
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/octet-stream",
    }
    
    file_size = os.path.getsize(filepath)
    print(f"⬆️  Uploading {filename} ({file_size / 1024 / 1024:.1f} MB)...")
    
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{upload_url_clean}?name={filename}",
            headers=headers,
            data=f,
            timeout=300,
        )
    
    if resp.status_code in (200, 201):
        data = resp.json()
        print(f"✅ {filename} berhasil diupload! URL: {data.get('browser_download_url', 'N/A')}")
        return data
    else:
        print(f"❌ Gagal upload {filename}: {resp.status_code} - {resp.text[:200]}")
        return None


def main():
    print("=" * 60)
    print("  GitHub Release Asset Uploader")
    print(f"  Repo: {GITHUB_REPO}")
    print(f"  Tag:  {RELEASE_TAG}")
    print("=" * 60)
    
    token = input("\nMasukkan GitHub Personal Access Token (dengan scope 'repo'): ").strip()
    if not token:
        print("❌ Token tidak boleh kosong!")
        sys.exit(1)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cek file tersedia
    for filename in FILES_TO_UPLOAD:
        fpath = os.path.join(project_dir, filename)
        if not os.path.exists(fpath):
            print(f"❌ File tidak ditemukan: {fpath}")
            sys.exit(1)
    
    # Buat release
    print(f"\n📦 Membuat/mencari GitHub Release '{RELEASE_TAG}'...")
    release = create_release(token)
    upload_url = release.get("upload_url", "")
    release_url = release.get("html_url", "")
    print(f"✅ Release: {release_url}")
    
    # Upload files
    print()
    for filename in FILES_TO_UPLOAD:
        fpath = os.path.join(project_dir, filename)
        upload_asset(token, upload_url, fpath, filename)
    
    print("\n" + "=" * 60)
    print("🎉 Selesai! File model tersedia di GitHub Release.")
    print(f"   URL Release: {release_url}")
    print("\n📝 Pastikan di app.py baris ini sudah benar:")
    print(f'   GITHUB_RELEASE_BASE = "https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"')

if __name__ == "__main__":
    main()
