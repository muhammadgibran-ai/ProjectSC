import zipfile
import os

def create_zip():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(project_dir, "Proyek_Sistem_Cerdas_Harga_Mobil.zip")
    
    files_to_pack = [
        "training.ipynb",
        "mobil_bekas_carmudi.csv",
        "model_harga_mobil.h5",
        "preprocessors.pkl",
        "app.py",
        "scraper.py",
        "train.py",
        "generate_report.py",
        "run_app.bat",
        "requirements.txt",
        "Laporan_Proyek_Sistem_Cerdas.docx"
    ]
    
    print(f"Creating ZIP archive at: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in files_to_pack:
            if f == "requirements.txt":
                file_path = os.path.join(project_dir, "requirements_submission.txt")
                if os.path.exists(file_path):
                    print("  Adding: requirements_submission.txt as requirements.txt")
                    zipf.write(file_path, arcname="requirements.txt")
                else:
                    std_path = os.path.join(project_dir, "requirements.txt")
                    if os.path.exists(std_path):
                        print("  Adding: requirements.txt")
                        zipf.write(std_path, arcname="requirements.txt")
            else:
                file_path = os.path.join(project_dir, f)
                if os.path.exists(file_path):
                    print(f"  Adding: {f}")
                    zipf.write(file_path, arcname=f)
                else:
                    print(f"  [Warning] File not found: {f}")
                
    print(f"ZIP archive successfully created: {zip_path}")

if __name__ == "__main__":
    create_zip()
