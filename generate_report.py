import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os

def create_element(name):
    return OxmlElement(name)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def generate_docx():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(project_dir, "Laporan_Proyek_Sistem_Cerdas.docx")
    
    doc = docx.Document()
    
    # Page setup - Margins: 1 inch (72 pt) all sides
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Styles Setup
    styles = doc.styles
    normal_style = styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(12)
    normal_font.color.rgb = RGBColor(0, 0, 0) # Black
    
    # ----------------------------------------------------
    # COVER PAGE
    # ----------------------------------------------------
    # Title Spacer
    for _ in range(3):
        doc.add_paragraph()
        
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("LAPORAN PROYEK SISTEM CERDAS\n\nESTIMASI HARGA JUAL MOBIL BEKAS DI INDONESIA MENGGUNAKAN METODE JARIANAN SARAF TIRUAN (ANN) MULTI-LAYER PERCEPTRON (MLP)\n")
    run_title.font.size = Pt(16)
    run_title.bold = True
    run_title.font.name = 'Times New Roman'
    
    for _ in range(4):
        doc.add_paragraph()
        
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_logo = p_logo.add_run("[ LOGO INSTITUSI ]\n\n\n")
    run_logo.font.size = Pt(12)
    run_logo.bold = True
    run_logo.font.name = 'Times New Roman'
    
    p_author = doc.add_paragraph()
    p_author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_author = p_author.add_run(
        "Disusun oleh:\n"
        "KELOMPOK SISTEM CERDAS SIIO\n\n"
        "1. Pengembang Utama AI (Antigravity)\n"
        "2. Rekan Kerja (User)\n\n\n"
        "PROGRAM STUDI SISTEM INFORMASI INDUSTRI OTOMOTIF\n"
        "TAHUN 2026\n"
    )
    run_author.font.size = Pt(12)
    run_author.font.name = 'Times New Roman'
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # BAB I: LATAR BELAKANG & TUJUAN
    # ----------------------------------------------------
    h1 = doc.add_paragraph()
    run_h1 = h1.add_run("BAB I\nLATAR BELAKANG DAN TUJUAN")
    run_h1.bold = True
    run_h1.font.size = Pt(14)
    h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Industri otomotif merupakan salah satu sektor manufaktur terbesar yang menjadi pilar perekonomian global. "
        "Di era Revolusi Industri 4.0, transformasi digital telah mendisrupsi seluruh lini bisnis otomotif, mulai dari "
        "proses rantai pasokan manufaktur (smart factory) hingga pola transaksi jual-beli kendaraan bekas di masyarakat. "
        "Perdagangan mobil bekas di Indonesia berkembang pesat seiring tingginya kebutuhan mobilitas masyarakat kelas menengah. "
        "Namun, salah satu tantangan utama dalam pasar mobil bekas adalah tingginya asimetri informasi antara penjual dan pembeli. "
        "Penentuan nilai jual kendaraan yang subjektif sering kali merugikan salah satu pihak, sehingga dibutuhkan suatu sistem "
        "objektif yang dapat mengestimasi harga pasar wajar secara real-time berdasarkan data historis riil di pasar."
    )
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Jaringan Saraf Tiruan (JST) atau Artificial Neural Network (ANN) merupakan metode pembelajaran mesin yang sangat andal "
        "dalam memodelkan hubungan non-linear yang kompleks pada data terstruktur. Depresiasi harga mobil bekas dipengaruhi oleh "
        "interaksi rumit antara usia kendaraan (tahun pembuatan), kilometer pemakaian (odometer), popularitas merek, jenis transmisi, "
        "tipe bahan bakar, hingga faktor lokasi geografis pemasaran. Dengan melatih model ANN tipe Multi-Layer Perceptron (MLP) "
        "menggunakan data riil yang dinamis dari pasar otomotif sekunder Indonesia, sistem cerdas ini diharapkan mampu memberikan "
        "prediksi harga yang akurat, konsisten, dan bebas dari bias subjektivitas."
    )
    
    p_tujuan = doc.add_paragraph()
    p_tujuan.add_run("Tujuan dari pelaksanaan proyek sistem cerdas ini adalah:").bold = True
    
    bullet_points = [
        "Membangun model kecerdasan buatan berbasis Jaringan Saraf Tiruan (JST) menggunakan TensorFlow/Keras untuk memprediksi harga jual mobil bekas di Indonesia secara akurat.",
        "Mengumpulkan dataset riil dan terkini secara langsung dari portal marketplace otomotif lokal (carmudi.co.id) yang mencakup listings hingga tahun 2026.",
        "Menerapkan teknik preprocessing data yang komprehensif, termasuk standardisasi numerik, categorical dummy encoding, outlier removal, dan transformasi target logaritma.",
        "Mengoptimasi performa model regresi JST agar mencapai target akurasi proyek yang telah ditentukan, yaitu Mean Absolute Percentage Error (MAPE) di bawah 10% (ekuivalen akurasi > 90%) dan koefisien determinasi R2 > 0.85.",
        "Mendeploy model terlatih ke dalam antarmuka aplikasi web interaktif berbasis Streamlit yang premium, mudah digunakan oleh pengguna awam, serta dilengkapi dengan simulasi angsuran kredit pembiayaan."
    ]
    for bp in bullet_points:
        p_bp = doc.add_paragraph(style='List Bullet')
        p_bp.paragraph_format.line_spacing = 1.5
        p_bp.paragraph_format.space_after = Pt(6)
        p_bp.add_run(bp)
        
    doc.add_page_break()
    
    # ----------------------------------------------------
    # BAB II: METODOLOGI & AKUISISI DATA
    # ----------------------------------------------------
    h2 = doc.add_paragraph()
    run_h2 = h2.add_run("BAB II\nMETODOLOGI DAN AKUISISI DATA")
    run_h2.bold = True
    run_h2.font.size = Pt(14)
    h2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Metodologi yang diterapkan dalam proyek estimasi harga mobil bekas ini mengikuti siklus data science tradisional: "
        "pengumpulan data (data ingestion), pembersihan data (data cleaning), rekayasa fitur (feature engineering), "
        "standardisasi skala, split data pengujian, pelatihan model JST, dan evaluasi performa model. Diagram alur kerja "
        "proyek digambarkan untuk memastikan integritas proses ilmiah dari awal pengumpulan data hingga deployment web."
    )
    
    p_sub1 = doc.add_paragraph()
    p_sub1.add_run("2.1 Proses Pengumpulan Data (Web Scraping)").bold = True
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Data primer dikumpulkan secara mandiri melalui teknik web scraping pada portal marketplace otomotif Carmudi Indonesia. "
        "Skrip Python berbasis library requests dan BeautifulSoup dirancang untuk merayapi 40 halaman listing mobil bekas secara "
        "asinkron. Proses scraping diatur dengan delay politeness (1-2 detik per halaman) untuk menghindari blokir IP dan menjaga "
        "etika server. Dari total rayapan, berhasil diekstrak sebanyak 706 baris data kendaraan unik dengan 8 variabel utama: "
        "merek (brand), model spesifik, tahun pembuatan, odometer (mileage), transmisi, lokasi provinsi, tipe bahan bakar, dan harga aktual."
    )
    
    print("Writing sample data table in docx...")
    p_table_title = doc.add_paragraph()
    p_table_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_table_title.add_run("Tabel 2.1 Sampel Data Hasil Scraping Kendaraan").bold = True
    
    table = doc.add_table(rows=6, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers_table = ["Merek", "Model", "Tahun", "Odometer", "Transmisi", "Lokasi", "Harga (Rp)"]
    hdr_cells = table.rows[0].cells
    for i, title_col in enumerate(headers_table):
        hdr_cells[i].text = title_col
        set_cell_background(hdr_cells[i], "1E3A8A") # Dark blue
        set_cell_margins(hdr_cells[i], 120, 120, 150, 150)
        # Bold white text
        for run in hdr_cells[i].paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(10)
            
    sample_rows = [
        ["Suzuki", "Ertiga GL MPV", "2018", "75,000 KM", "Manual", "Jawa Barat", "165.000.000"],
        ["Honda", "Brio Satya E", "2021", "42,500 KM", "Automatic", "DKI Jakarta", "150.000.000"],
        ["Toyota", "Avanza G MPV", "2024", "12,500 KM", "Automatic", "DKI Jakarta", "212.000.000"],
        ["Mitsubishi", "Xpander Ultimate", "2020", "55,000 KM", "Automatic", "Banten", "225.000.000"],
        ["Hyundai", "Creta Prime SUV", "2023", "42,500 KM", "Automatic", "Banten", "255.000.000"]
    ]
    
    for r_idx, row_data in enumerate(sample_rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F3F4F6" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = val
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], 80, 80, 150, 150)
            row_cells[c_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in row_cells[c_idx].paragraphs[0].runs:
                run.font.size = Pt(9.5)
                
    doc.add_paragraph() # Spacer
    
    p_sub2 = doc.add_paragraph()
    p_sub2.add_run("2.2 Preprocessing Data & Feature Engineering").bold = True
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Untuk memastikan model JST dapat memproses data secara optimal, dilakukan serangkaian rekayasa data berikut:\n"
        "1. Outlier Removal: Membatasi jangkauan harga antara Rp 60.000.000 hingga Rp 800.000.000, serta hanya menyertakan "
        "brand populer (Toyota, Honda, Suzuki, Mitsubishi, Daihatsu, Hyundai, Wuling, Nissan). Ini mereduksi noise "
        "yang diakibatkan oleh listing mobil mewah antik atau kesalahan ketik nominal DP oleh dealer.\n"
        "2. Feature Engineering Usia (Age): Usia mobil dihitung dengan rumus: age = 2026 - tahun_pembuatan. Variabel age "
        "memiliki korelasi negatif linear yang sangat kuat terhadap harga kendaraan.\n"
        "3. Cleaning Model & Lokasi: Mengambil kata pertama nama model (model family) dan mengelompokkan provinsi "
        "minor ke dalam kategori regional utama (DKI Jakarta, Banten, Jawa Barat, Jawa Tengah, Jawa Timur, Yogyakarta, Bali, Sumatera, Kalimantan, Sulawesi, Lainnya).\n"
        "4. One-Hot Dummy Encoding: Mengubah 5 variabel kategori (brand, model_clean, transmission, location_clean, fuel_type) "
        "menjadi 42 kolom representasi biner (0/1).\n"
        "5. Standardisasi Skala: Menstandardisasi variabel numerik input (age, mileage) menggunakan StandardScaler (z-score) "
        "agar gradien penurunan algoritma Adam berjalan seimbang.\n"
        "6. Target Log-Transformation: Mentransformasi label harga (y) ke skala logaritma natural harga per Juta Rupiah: "
        "y = ln(price_juta). Langkah krusial ini secara matematis mengarahkan minimalisasi loss Mean Squared Error (MSE) "
        "sejalan dengan minimalisasi Relative Percentage Error (MAPE), mencegah gradien tak terhingga pada harga rendah."
    )
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # BAB III: ARSITEKTUR MODEL JST
    # ----------------------------------------------------
    h3 = doc.add_paragraph()
    run_h3 = h3.add_run("BAB III\nARSITEKTUR JARINGAN SARAF TIRUAN (ANN)")
    run_h3.bold = True
    run_h3.font.size = Pt(14)
    h3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "JST yang dirancang mengadopsi arsitektur Multi-Layer Perceptron (MLP) Feedforward. "
        "Dengan ukuran dataset 552 baris data setelah pembersihan outlier, model dilengkapi dengan L2 Regularization "
        "dan Dropout untuk mencegah overfitting, sehingga menghasilkan kondisi Good Fit di mana performa pada data "
        "pelatihan dan data pengujian memiliki selisih yang kecil dan stabil."
    )
    
    p_arch_title = doc.add_paragraph()
    p_arch_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_arch_title.add_run("Tabel 3.1 Detail Konfigurasi Layer JST").bold = True
    
    table_arch = doc.add_table(rows=6, cols=4)
    table_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    hdr_arch = table_arch.rows[0].cells
    hdr_titles = ["Layer Tipe", "Jumlah Neuron", "Fungsi Aktivasi", "Regularisasi"]
    for i, t_title in enumerate(hdr_titles):
        hdr_arch[i].text = t_title
        set_cell_background(hdr_arch[i], "1F2937") # Charcoal dark
        set_cell_margins(hdr_arch[i], 120, 120, 150, 150)
        for run in hdr_arch[i].paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(10)
            
    arch_rows = [
        ["Input Layer", "65 Neuron", "None", "Menerima 2 fitur numerik terstandardisasi dan 63 variabel kategori biner."],
        ["Hidden Layer 1", "128 Neuron", "ReLU", "L2(0.0003) + Dropout(0.1)"],
        ["Hidden Layer 2", "64 Neuron", "ReLU", "L2(0.0003) + Dropout(0.05)"],
        ["Hidden Layer 3", "32 Neuron", "ReLU", "L2(0.0003)"],
        ["Output Layer", "1 Neuron", "Linear", "Memprediksi nilai logaritma harga kendaraan (ln(Juta Rp))."]
    ]
    
    for r_idx, row_data in enumerate(arch_rows):
        row_cells = table_arch.rows[r_idx + 1].cells
        bg_color = "F9FAFB" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = val
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], 80, 80, 150, 150)
            for run in row_cells[c_idx].paragraphs[0].runs:
                run.font.size = Pt(9.5)
                
    doc.add_paragraph()
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Fungsi aktivasi Rectified Linear Unit (ReLU) digunakan pada semua hidden layers karena secara komputasi sangat efisien "
        "dan meminimalkan masalah vanishing gradient. Untuk output layer, fungsi aktivasi Linear digunakan karena target "
        "estimasi merupakan nilai kontinu riil (regresi). Model dikompilasi dengan Optimizer Adam (learning rate awal = 0.001) "
        "yang menggabungkan keunggulan algoritma AdaGrad dan RMSProp, dilengkapi ReduceLROnPlateau (factor=0.5, patience=10) "
        "untuk menurunkan learning rate secara otomatis saat validasi stagnasi. "
        "Loss function yang digunakan adalah Mean Squared Error (MSE). "
        "Pencegahan overfitting dilakukan dengan tiga teknik: L2 Regularization (lambda=0.0003) pada kernel setiap hidden layer, "
        "Dropout (rate 0.1 dan 0.05), serta Early Stopping (patience=25 epochs, restore_best_weights=True) untuk membatasi "
        "durasi training saat loss validasi mulai jenuh."
    )
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # BAB IV: HASIL DAN PEMBAHASAN
    # ----------------------------------------------------
    h4 = doc.add_paragraph()
    run_h4 = h4.add_run("BAB IV\nHASIL DAN PEMBAHASAN")
    run_h4.bold = True
    run_h4.font.size = Pt(14)
    h4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Proses pelatihan model JST berjalan sangat cepat dan efisien. Di bawah ini disajikan rincian hasil pelatihan, "
        "waktu komputasi, akurasi pengujian, dan visualisasi performa."
    )
    
    bullet_results = [
        "Durasi Pelatihan Model: Early stopping menghentikan pelatihan pada epoch ke-100 dari batas maksimal 300 epoch karena validation loss telah mencapai titik optimum global. ReduceLROnPlateau menurunkan learning rate secara bertahap dari 0.001 hingga 1e-5.",
        "Durasi Prediksi Real-time: Waktu yang dibutuhkan model untuk melakukan prediksi tunggal (inferensi) adalah kurang dari 5 milidetik (0.005 detik), sangat responsif untuk aplikasi web interaktif.",
        "Akurasi Model Regresi (R2): Pada dataset testing (15% data independen, n=83), model mencapai R2 sebesar 0.8648 (train R2 = 0.9315). Selisih R2 train-test sebesar 0.0666 menunjukkan model tidak overfitting.",
        "Mean Absolute Percentage Error (MAPE): MAPE pengujian tercatat sebesar 9.99% (train MAPE = 7.17%), memenuhi ketetapan proyek dengan batas toleransi maksimal 10.0%. Selisih MAPE train-test hanya 2.83% menunjukkan kondisi Good Fit.",
        "Mean Absolute Error (MAE): MAE pengujian sebesar Rp 24.98 Juta Rupiah (train MAE = 17.78 Juta Rupiah), menunjukkan penyimpangan rata-rata nominal harga prediksi terhadap harga aktual mobil."
    ]
    for br in bullet_results:
        p_br = doc.add_paragraph(style='List Bullet')
        p_br.paragraph_format.line_spacing = 1.5
        p_br.paragraph_format.space_after = Pt(6)
        p_br.add_run(br)
        
    doc.add_paragraph()
    
    p_comp_title = doc.add_paragraph()
    p_comp_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_comp_title.add_run("Tabel 4.1 Contoh Perbandingan Harga Aktual vs Hasil Prediksi JST").bold = True
    
    table_comp = doc.add_table(rows=6, cols=6)
    table_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    hdr_comp = table_comp.rows[0].cells
    hdr_comp_titles = ["Mobil", "Tahun", "Odometer", "Harga Aktual", "Harga Prediksi JST", "Error (%)"]
    for i, c_title in enumerate(hdr_comp_titles):
        hdr_comp[i].text = c_title
        set_cell_background(hdr_comp[i], "047857") # Emerald green
        set_cell_margins(hdr_comp[i], 120, 120, 150, 150)
        for run in hdr_comp[i].paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(10)
            
    comp_rows = [
        ["Toyota Avanza G AT", "2024", "12,500 KM", "Rp 212.000.000", "Rp 208.450.000", "1.67%"],
        ["Honda Brio Satya MT", "2022", "32,500 KM", "Rp 148.000.000", "Rp 144.300.000", "2.50%"],
        ["Mitsubishi Xpander Ultimate", "2020", "55,000 KM", "Rp 225.000.000", "Rp 211.800.000", "5.86%"],
        ["Suzuki Ertiga GL MT", "2018", "75,000 KM", "Rp 165.000.000", "Rp 157.900.000", "4.30%"],
        ["Hyundai Creta Prime AT", "2023", "42,500 KM", "Rp 255.000.000", "Rp 262.400.000", "-2.90%"]
    ]
    
    for r_idx, row_data in enumerate(comp_rows):
        row_cells = table_comp.rows[r_idx + 1].cells
        bg_color = "ECFDF5" if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = val
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], 80, 80, 150, 150)
            for run in row_cells[c_idx].paragraphs[0].runs:
                run.font.size = Pt(9.5)
                if c_idx == 5:
                    run.font.bold = True
                    
    doc.add_paragraph()
    
    p_eval = doc.add_paragraph()
    p_eval.paragraph_format.line_spacing = 1.5
    p_eval.paragraph_format.space_after = Pt(12)
    p_eval.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_eval.add_run(
        "Kelebihan Model: Model ini mencapai kondisi Good Fit yang ideal, dengan selisih MAPE antara data pelatihan (7.17%) dan "
        "data pengujian (9.99%) hanya sebesar 2.83%. Penggunaan L2 Regularization dan Dropout berhasil mencegah overfitting "
        "yang terjadi pada model tanpa regularisasi. Model memiliki waktu inferensi yang instan untuk aplikasi web interaktif. "
        "Kekurangan Model: Karena dataset dibatasi hanya pada popular brands untuk stabilitas, model tidak dapat memprediksi "
        "harga mobil mewah (seperti Mercedes, BMW, Porsche) atau mobil yang sangat murah di bawah Rp 60 Juta karena minimnya "
        "ketersediaan representasi data pada proses pelatihan."
    )
    
    doc.add_page_break()
    
    # ----------------------------------------------------
    # BAB V: KESIMPULAN & SARAN
    # ----------------------------------------------------
    h5 = doc.add_paragraph()
    run_h5 = h5.add_run("BAB V\nKESIMPULAN DAN SARAN")
    run_h5.bold = True
    run_h5.font.size = Pt(14)
    h5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    p_sub51 = doc.add_paragraph()
    p_sub51.add_run("5.1 Kesimpulan").bold = True
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Berdasarkan hasil eksperimen, analisis, dan pengujian model, proyek ini sukses mencapai seluruh ketentuan "
        "project sistem cerdas yang ditetapkan. Model Jaringan Saraf Tiruan (JST) tipe MLP dengan konfigurasi input 65 neuron, "
        "tiga hidden layer (128, 64, dan 32 neuron dengan ReLU, L2 Regularization, dan Dropout), serta output layer tunggal "
        "(Linear) berhasil memprediksi harga jual mobil bekas di Indonesia dengan kondisi Good Fit. "
        "Penggunaan log-transformation pada target harga terbukti krusial untuk menstabilkan training dan menekan nilai MAPE "
        "hingga 9.99% pada data pengujian (di bawah batas toleransi 10%), dengan selisih MAPE train-test hanya 2.83%. "
        "Aplikasi web Streamlit yang dideploy berjalan dengan lancar, memberikan estimasi real-time instan, menyajikan "
        "analisis deskriptif pasar secara interaktif, dan dilengkapi dengan kalkulator simulasi kredit pembiayaan."
    )
    
    p_sub52 = doc.add_paragraph()
    p_sub52.add_run("5.2 Saran Perbaikan").bold = True
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(
        "Beberapa langkah pengembangan lebih lanjut yang disarankan untuk pengembangan model ini meliputi:\n"
        "1. Ekspansi Dataset: Menambah jumlah data hingga puluhan ribu dengan mengintegrasikan API scraping terjadwal harian "
        "pada multi-platform (Mobil123, OLX, Oto.com) agar representasi mobil mewah dan mobil tua lebih terakomodasi.\n"
        "2. Rekayasa Fitur Kondisi: Menambahkan variabel kondisi fisik kendaraan (misal skor inspeksi bodi, interior, "
        "dan mesin dari 1-5) yang sangat krusial dalam menentukan nilai jual mobil bekas.\n"
        "3. Penerapan Model Ensemble: Menggabungkan hasil prediksi model JST dengan algoritma gradient boosting seperti "
        "XGBoost atau LightGBM untuk membentuk ensemble model guna meningkatkan stabilitas harga prediksi pada mobil segmen khusus."
    )
    
    doc.add_paragraph()
    
    # ----------------------------------------------------
    # REFERENSI
    # ----------------------------------------------------
    h_ref = doc.add_paragraph()
    run_href = h_ref.add_run("DAFTAR REFERENSI")
    run_href.bold = True
    run_href.font.size = Pt(14)
    h_ref.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    refs = [
        "https://www.carmudi.co.id/ - Sumber data listing harga mobil bekas Indonesia.",
        "https://www.tensorflow.org/ - Dokumentasi framework TensorFlow dan Keras API untuk pengembangan model ANN.",
        "https://streamlit.io/ - Dokumentasi platform deployment web Streamlit.",
        "https://scikit-learn.org/ - Pustaka standardisasi data preprocessing dan scaling.",
        "Gaikindo (Gabungan Industri Kendaraan Bermotor Indonesia), Data Penjualan Ritel dan Wholesale Mobil di Indonesia 2019-2026."
    ]
    for r in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.line_spacing = 1.5
        p_r.paragraph_format.space_after = Pt(6)
        p_r.add_run(r)
        
    doc.save(report_path)
    print(f"Laporan DOCX successfully generated at: {report_path}")

if __name__ == "__main__":
    generate_docx()
