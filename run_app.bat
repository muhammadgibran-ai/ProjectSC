@echo off
title Aplikasi Estimasi Harga Mobil Bekas - Sistem Cerdas
echo ==========================================================
echo       MENJALANKAN APLIKASI WEB ESTIMASI HARGA MOBIL
echo ==========================================================
echo.
echo Sedang meluncurkan server lokal Streamlit...
echo.
streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Gagal menjalankan Streamlit. Pastikan library sudah terinstall.
    echo Silakan jalankan 'pip install -r requirements.txt' terlebih dahulu.
    echo.
    pause
)
