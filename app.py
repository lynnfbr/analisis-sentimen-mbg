"""
Streamlit App - Perbandingan Algoritma Naive Bayes dan Support Vector Machine untuk 
Analisis Sentimen Masyarakat terhadap Program MBG pada Media Sosial X
Jalankan dengan: streamlit run app.py

Struktur folder:
    app.py
    preprocessing.py
    requirements.txt
    models/  (tfidf_vectorizer.pkl, naive_bayes_model.pkl, svm_model.pkl)
    data/    (dataset_filtered.csv, dataset_labeled_binary.csv, dataset_labeled_3class.csv)
"""

import os
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from preprocessing import preprocess_text, preprocess_with_steps
    PREPROCESSING_AVAILABLE = True
except Exception as e:
    PREPROCESSING_AVAILABLE = False
    PREPROCESSING_IMPORT_ERROR = str(e)

# =====================================================================================
# KONFIGURASI HALAMAN
# =====================================================================================
st.set_page_config(
    page_title="Sentimen MBG - NB vs SVM",
    page_icon="🍱",
    layout="wide",
)

MODEL_DIR = "models"
LABEL_MAP = {0: "Negatif", 1: "Positif"}  # sudah dikonfirmasi dari dataset_labeled_binary.csv

DATA_FILTERED_PATH = os.path.join("data", "dataset_filtered.csv")
DATA_LABELED_BINARY_PATH = os.path.join("data", "dataset_labeled_binary.csv")
DATA_LABELED_3CLASS_PATH = os.path.join("data", "dataset_labeled_3class.csv")

COLORS = {
    "primary": "#A7C7E8",     # pastel blue
    "secondary": "#B8E3D4",   # pastel mint
    "dark": "#5C6B73",        # slate lembut, dipakai utk aksen non-teks (border, dsb)
    "negatif": "#F2A7A0",     # pastel coral
    "positif": "#9FD8C4",     # pastel teal-green
    "netral": "#C9CDD3",      # pastel abu-abu
    "bg": "#FAF9FB",
}

# =====================================================================================
# CSS
# =====================================================================================

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .material-symbols-outlined {{
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }}

    /* Sembunyikan header bawaan Streamlit (bar berisi tombol Deploy / menu ≡).
       Bar ini position:fixed dengan z-index sangat tinggi, jadi tadinya
       nutupin sebagian besar navbar custom kita di atas. */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    div[data-testid="stAppViewContainer"] {{
        padding-top: 0 !important;
    }}

    /* Batasi lebar konten biar rapi di layar lebar & gambar/plot tidak
       kepotong atau melar aneh */
    .block-container {{
        max-width: 1180px;
        margin: 0 auto;
        padding-top: 0.6rem;
        padding-bottom: 3rem;
    }}
    img, .stImage img, [data-testid="stImage"] img {{
        max-width: 100%;
        height: auto;
    }}

    /* Sidebar tidak dipakai lagi — navigasi dipindah ke topbar */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* ===== TOPBAR / NAVBAR (gaya landing page) ===== */
    .st-key-navbar {{
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 9999;
        background: var(--background-color);
        border-bottom: 1px solid rgba(128,128,128,0.12);
        padding: 0.7rem 0 0.8rem 0;
        margin: 0 0 1.6rem 0;
    }}
    .navbar-brand {{
        font-weight: 800;
        font-size: 1.15rem;
        color: var(--text-color);
        display: flex;
        align-items: center;
        height: 100%;
        gap: 0.5rem;
        white-space: nowrap;
    }}
    .navbar-brand .material-symbols-outlined {{
        color: {COLORS['primary']};
        font-size: 1.5rem;
        flex-shrink: 0;
    }}
    .navbar-tagline {{
        font-size: 0.7rem;
        opacity: 0.55;
        font-weight: 400;
        margin-top: -2px;
        white-space: nowrap;
    }}

    /* Tombol nav di dalam navbar — dibuat kayak link, satu baris, tidak wrap */
    .st-key-navbar .stButton {{
        display: flex;
        align-items: center;
        height: 100%;
    }}
    .st-key-navbar .stButton>button {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-color);
        padding: 0.5rem 0.6rem;
        box-shadow: none;
        white-space: nowrap;
        width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .st-key-navbar .stButton>button p {{
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .st-key-navbar .stButton>button:hover {{
        background: rgba(167,199,232,0.16);
        transform: none;
        box-shadow: none;
        border-color: transparent;
    }}
    .st-key-navbar .stButton>button[kind="primary"] {{
        background: {COLORS['primary']} !important;
        color: #2D3436 !important;
        border: none !important;
    }}

    /* Tombol CTA "Try Now" — dibungkus container key sendiri biar bisa
       di-style beda (pill, selalu solid) dari menu nav lainnya */
    .st-key-navcta .stButton>button {{
        border-radius: 20px !important;
        font-weight: 700 !important;
        background: {COLORS['primary']} !important;
        color: #2D3436 !important;
        border: none !important;
        padding: 0.5rem 1.1rem !important;
    }}
    .st-key-navcta .stButton>button:hover {{
        filter: brightness(0.95);
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    }}

    /* Hero banner — background pastel eksplisit, teks gelap supaya kontras tetap
       terjaga di kedua tema */
    .hero {{
        background: linear-gradient(120deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        color: #2D3436;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }}
    .hero h1 {{ margin: 0 0 0.4rem 0; font-weight: 800; color: #2D3436; }}
    .hero p {{ margin: 0; opacity: 0.85; font-size: 1.02rem; color: #2D3436; }}

    /* Card — ikut warna tema aktif (light/dark) via CSS variable Streamlit,
       jadi otomatis kebaca di kedua mode */
    .card {{
        background: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 2px 14px rgba(0,0,0,0.06);
        border: 1px solid rgba(128,128,128,0.15);
        transition: all 0.25s ease;
        height: 100%;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 14px 28px rgba(0,0,0,0.12);
        border-color: {COLORS['primary']}aa;
    }}
    .card h4 {{ margin-top: 0; color: var(--text-color); }}
    .card p {{ color: var(--text-color); }}
    .card-icon {{ font-size: 1.8rem; margin-bottom: 0.4rem; }}
    .card .muted {{ opacity: 0.62; }}

    /* Pengganti emoji: badge angka bulat buat step/urutan */
    .step-num {{
        width: 30px; height: 30px; border-radius: 50%;
        background: {COLORS['primary']}; color: #2D3436;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; margin-bottom: 0.5rem;
    }}

    /* Pengganti emoji: chip label kecil (NB / SVM, dsb) */
    .chip {{
        display: inline-block; padding: 3px 12px; border-radius: 8px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
        color: #2D3436; margin-bottom: 0.5rem;
    }}
    .chip-nb {{ background: {COLORS['primary']}; }}
    .chip-svm {{ background: {COLORS['secondary']}; }}

    /* Section title */
    .section-title {{
        font-weight: 700;
        font-size: 1.3rem;
        color: var(--text-color);
        border-left: 5px solid {COLORS['primary']};
        padding-left: 0.7rem;
        margin: 1.4rem 0 0.9rem 0;
    }}

    /* Buttons umum (bukan di navbar) */
    .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }}
    .stButton>button[kind="primary"] {{
        background: linear-gradient(120deg, {COLORS['primary']}, {COLORS['secondary']});
        color: #2D3436;
        border: none;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: var(--secondary-background-color);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        border: 1px solid rgba(128,128,128,0.15);
    }}
    [data-testid="stMetric"]:hover {{ transform: translateY(-3px); }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 10px 22px;
        font-weight: 600;
        background: var(--secondary-background-color);
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']} !important;
        color: #2D3436 !important;
    }}

    /* Result badge — bg+teks eksplisit, sengaja tidak ikut tema supaya
       makna warna (benar/salah) tetap konsisten */
    .badge-ok {{
        background: #DCF3EA; color: #1E7A5F; padding: 6px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }}
    .badge-bad {{
        background: #FBE4E1; color: #B0413E; padding: 6px 14px;
        border-radius: 20px; font-weight: 600; display: inline-block;
    }}

    /* Dataframe row hover — overlay tint tipis, aman di kedua tema */
    [data-testid="stDataFrame"] div[role="row"]:hover {{
        background-color: rgba(167,199,232,0.18) !important;
    }}

    /* ================= SCROLL-REVEAL ANIMATION ================= */
    /* PENTING: opacity default harus 1 (terlihat). Animasi geser hanya
       aktif kalau JS berhasil jalan (browser mendukung + koneksi cukup
       cepat). Ini mencegah judul/konten "hilang" kalau script di
       components.html gagal jalan (mis. karena sandboxing iframe atau
       koneksi lambat) -- dulu opacity default 0, jadi kalau JS gagal,
       elemen permanen tak terlihat sampai fallback 4 detik (atau
       selamanya kalau fallback pun gagal jalan). */
    .reveal-left, .reveal-right, .reveal-up {{
        opacity: 1;
        transition: opacity 0.5s ease, transform 0.5s ease;
        will-change: opacity, transform;
    }}
    .reveal-pending.reveal-left {{ opacity: 0; transform: translateX(-45px); }}
    .reveal-pending.reveal-right {{ opacity: 0; transform: translateX(45px); }}
    .reveal-pending.reveal-up {{ opacity: 0; transform: translateY(28px); }}
    .reveal-visible {{ opacity: 1 !important; transform: translate(0, 0) !important; }}
    .reveal-d1 {{ transition-delay: 0.08s; }}
    .reveal-d2 {{ transition-delay: 0.16s; }}
    .reveal-d3 {{ transition-delay: 0.24s; }}
    .reveal-d4 {{ transition-delay: 0.32s; }}
    .reveal-d5 {{ transition-delay: 0.40s; }}
    .reveal-d6 {{ transition-delay: 0.48s; }}

    /* ================= HERO (split, gaya landing page) ================= */
    .pill-badge {{
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        background: {COLORS['secondary']}55;
        color: var(--text-color);
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.03em;
        margin-bottom: 0.9rem;
    }}
    .hero-split h1 {{
        font-weight: 800;
        font-size: 2.2rem;
        line-height: 1.18;
        margin: 0 0 0.9rem 0;
        color: var(--text-color);
    }}
    .hero-split h1 .accent {{ color: {COLORS['primary']}; }}
    .hero-split p.hero-desc {{
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.3rem;
        max-width: 480px;
    }}
    .hero-stats {{ display: flex; gap: 1.8rem; margin-top: 1.6rem; }}
    .hero-stats .stat-num {{ font-weight: 800; font-size: 1.5rem; color: var(--text-color); }}
    .hero-stats .stat-label {{ font-size: 0.7rem; opacity: 0.6; }}

    .preview-card {{
        background: var(--secondary-background-color);
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 14px 32px rgba(0,0,0,0.10);
        border: 1px solid rgba(128,128,128,0.15);
    }}
    .preview-card .preview-title {{
        font-weight: 700;
        font-size: 0.85rem;
        opacity: 0.6;
        margin-bottom: 0.6rem;
    }}
    .preview-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        padding: 0.65rem 0;
        border-bottom: 1px dashed rgba(128,128,128,0.18);
    }}
    .preview-row:last-child {{ border-bottom: none; }}
    .preview-text {{ font-size: 0.82rem; color: var(--text-color); opacity: 0.85; }}
    .sent-badge {{
        flex-shrink: 0;
        padding: 3px 11px;
        border-radius: 12px;
        font-size: 0.68rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .sent-badge-pos {{ background: {COLORS['positif']}44; color: #1E7A5F; }}
    .sent-badge-neg {{ background: {COLORS['negatif']}44; color: #B0413E; }}

    /* ================= ALUR / FLOW (numbered, dengan panah) ================= */
    .flow-row {{ display: flex; align-items: flex-start; gap: 0.3rem; flex-wrap: wrap; }}
    .flow-item {{ flex: 1; min-width: 118px; text-align: center; }}
    .flow-item .step-num {{ margin: 0 auto 0.5rem auto; }}
    .flow-item h4 {{ font-size: 0.9rem; margin: 0 0 0.25rem 0; color: var(--text-color); }}
    .flow-item p {{ font-size: 0.74rem; opacity: 0.6; margin: 0; }}
    .flow-arrow {{
        flex: 0 0 auto;
        align-self: center;
        margin-top: -2.2rem;
        opacity: 0.35;
        font-size: 1.3rem;
    }}

    /* ================= FEATURE ROWS ("Mengapa") ================= */
    .feature-row {{ display: flex; gap: 0.9rem; align-items: flex-start; margin-bottom: 1.3rem; }}
    .feature-icon-box {{
        flex-shrink: 0;
        width: 42px; height: 42px;
        border-radius: 12px;
        background: {COLORS['primary']}33;
        display: flex; align-items: center; justify-content: center;
    }}
    .feature-icon-box .material-symbols-outlined {{ color: {COLORS['primary']}; font-size: 1.4rem; }}
    .feature-row h4 {{ margin: 0 0 0.2rem 0; font-size: 0.95rem; color: var(--text-color); }}
    .feature-row p {{ margin: 0; font-size: 0.82rem; opacity: 0.65; }}

    /* Stat card (bar progres sebaran sentimen) */
    .stat-card-title {{ font-weight: 700; font-size: 0.95rem; margin-bottom: 0.9rem; color: var(--text-color); }}
    .stat-bar-label {{ display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 3px; color: var(--text-color); }}
    .stat-bar-track {{ background: rgba(128,128,128,0.15); border-radius: 8px; height: 8px; overflow: hidden; margin-bottom: 12px; }}
    .stat-bar-fill {{ height: 100%; border-radius: 8px; }}
    .mini-metric {{
        border-radius: 12px;
        padding: 0.6rem 0.5rem;
        text-align: center;
    }}
    .mini-metric .num {{ font-weight: 800; font-size: 1.15rem; color: var(--text-color); }}
    .mini-metric .label {{ font-size: 0.65rem; opacity: 0.65; color: var(--text-color); }}

    /* ================= TECH STACK ICONS ================= */
    .tech-icon-box {{
        width: 52px; height: 52px;
        border-radius: 14px;
        background: {COLORS['primary']};
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.6rem auto;
        box-shadow: 0 6px 14px {COLORS['primary']}55;
    }}
    .tech-icon-box .material-symbols-outlined {{ color: #2D3436; font-size: 1.6rem; }}
    .tech-item {{ text-align: center; }}
    .tech-item h4 {{ font-size: 0.88rem; margin: 0 0 0.15rem 0; color: var(--text-color); }}
    .tech-item p {{ font-size: 0.72rem; opacity: 0.6; margin: 0; }}

    /* Checklist di kartu perbandingan model */
    .check-item {{
        display: flex; align-items: center; gap: 6px;
        font-size: 0.83rem; margin: 6px 0; color: var(--text-color);
    }}
    .check-item .material-symbols-outlined {{ color: {COLORS['positif']}; font-size: 1.05rem; }}
    .accuracy-banner {{
        border-radius: 12px;
        text-align: center;
        padding: 0.7rem;
        margin-top: 1rem;
    }}
    .accuracy-banner .num {{ font-size: 1.6rem; font-weight: 800; color: #2D3436; }}
    .accuracy-banner .label {{ font-size: 0.68rem; opacity: 0.75; color: #2D3436; }}

    /* CTA banner penutup halaman Beranda */
    .cta-banner {{
        background: linear-gradient(120deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        text-align: center;
        color: #2D3436;
        margin-top: 0.6rem;
    }}
    .cta-banner h3 {{ margin: 0 0 0.4rem 0; font-weight: 800; }}
    .cta-banner p {{ margin: 0 0 1.1rem 0; opacity: 0.8; }}
    .st-key-cta_banner_btn .stButton>button {{
        border-radius: 24px !important;
        padding: 0.65rem 1.8rem !important;
        font-weight: 700 !important;
        background: #2D3436 !important;
        color: white !important;
        border: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def inject_scroll_reveal():
    """Suntik IntersectionObserver ke DOM utama Streamlit (lewat window.parent)
    supaya elemen ber-class reveal-left/reveal-right/reveal-up mendapat animasi
    geser saat discroll ke posisinya.

    PENTING (progressive enhancement): elemen-elemen itu defaultnya SUDAH
    terlihat (lihat CSS .reveal-left/.reveal-right/.reveal-up -> opacity:1).
    Script ini baru menambahkan class 'reveal-pending' (yang men-transparankan
    elemen) TEPAT SEBELUM mulai mengamati elemen tsb. Jadi kalau script ini
    gagal jalan sama sekali (mis. iframe di-sandbox oleh browser/proxy saat
    deploy), elemen tetap terlihat normal alih-alih hilang permanen -- beda
    dengan versi sebelumnya yang defaultnya opacity:0 dan bergantung penuh
    pada JS/timer fallback untuk memunculkannya."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function activate() {
                const els = doc.querySelectorAll('.reveal-left, .reveal-right, .reveal-up');
                if (!els.length) return;
                els.forEach((el) => el.classList.add('reveal-pending'));
                const io = new IntersectionObserver((entries) => {
                    entries.forEach((entry) => {
                        if (entry.isIntersecting) {
                            entry.target.classList.add('reveal-visible');
                            io.unobserve(entry.target);
                        }
                    });
                }, { threshold: 0.12, rootMargin: '0px 0px -60px 0px' });
                els.forEach((el) => io.observe(el));
                // Fallback: kalau ada elemen yang lolos observer, paksa tampil.
                setTimeout(() => {
                    els.forEach((el) => el.classList.add('reveal-visible'));
                }, 2500);
            }
            try {
                activate();
                setTimeout(activate, 350);
            } catch (e) {
                // JS gagal total -> elemen tetap terlihat (opacity:1 default),
                // cukup tidak dapat animasi geser. Tidak perlu tindakan lain.
            }
        })();
        </script>
        """,
        height=0,
    )


inject_css()

# =====================================================================================
# DATA HASIL EKSPERIMEN (dari notebook Colab)
#
# CATATAN: proses pengurangan data terdiri atas 2 tahap yang BERBEDA secara
# metodologis (mengikuti Bab III subbab 3.7.3.2 dan 3.7.3.4):
#   Tahap 1 - FILTERING: menyaring tweet mentah hasil scraping (bukan bahasa
#             Indonesia, akun media, template campaign, meme, hashtag
#             campaign, frasa mobilisasi, dst). 30.692 -> 21.231.
#   Tahap 2 - PEMERIKSAAN KUALITAS PASCA-PREPROCESSING: dilakukan SETELAH
#             text preprocessing (bukan bagian dari filtering tahap 1),
#             meliputi penghapusan hasil kosong, duplikat hasil
#             preprocessing, teks terlalu pendek, pola campaign lanjutan,
#             dan tweet dengan kata < 3. 21.231 -> 19.585.
# =====================================================================================

FILTERING_FUNNEL = {
    "Tahap": [
        "Data Awal (Scraping)", "Hapus Duplikat (Raw Text)", "Hapus Bukan Bahasa Indonesia",
        "Hapus Akun Media/Berita", "Hapus Tweet Tanpa Opini\n(Gaya Berita)", "Hapus Template Campaign Berulang",
        "Hapus Meme/Konten Promosi", "Hapus Hashtag Campaign\nTerkoordinasi", "Hapus Frasa Mobilisasi",
    ],
    "Jumlah Data": [30692, 24726, 23908, 23073, 22243, 22155, 21815, 21435, 21231],
}

QUALITY_CHECK = {
    "before": 21231,
    "after": 19585,
    "steps": [
        "Menghapus data yang menjadi kosong setelah proses cleaning",
        "Menghapus duplikat berdasarkan hasil preprocessing (bukan raw text)",
        "Menghapus tweet dengan panjang teks hasil preprocessing yang terlalu pendek",
        "Penyaringan pola campaign lanjutan (final campaign pattern filtering)",
        "Menghapus tweet dengan jumlah kata kurang dari 3",
    ],
}

# Dipertahankan untuk kompatibilitas kalau ada bagian lain yang masih
# menghitung total keseluruhan (scraping -> siap dilabeli).
FUNNEL_DATA = {
    "Tahap": FILTERING_FUNNEL["Tahap"] + ["Pemeriksaan Kualitas\nPasca-Preprocessing\n(Data Final Labeling)"],
    "Jumlah Data": FILTERING_FUNNEL["Jumlah Data"] + [QUALITY_CHECK["after"]],
}

SENTIMENT_DIST = {
    "Sentimen": ["Negatif", "Positif", "Netral"],
    "Jumlah": [12540, 6298, 747],
    "Persentase": [64.03, 32.16, 3.81],
}

SPLIT_DATA = {"Training (80%)": 15070, "Testing (20%)": 3768}

NB_METRICS = {"Accuracy": 79.38, "Precision": 76.98, "Recall": 54.68, "F1-Score": 63.94}
SVM_METRICS = {"Accuracy": 88.64, "Precision": 84.27, "Recall": 81.19, "F1-Score": 82.70}

NB_TUNING = pd.DataFrame({
    "Alpha": [0.1, 0.5, 1.0, 2.0],
    "Mean F1 Score": [0.626009, 0.624461, 0.617606, 0.595251],
    "Ranking": [1, 2, 3, 4],
})
NB_BEST_PARAM = "alpha = 0.1"
NB_TRAIN_TIME = 3.4344

SVM_TUNING = pd.DataFrame({
    "Nilai C": [1.00, 5.00, 10.00, 20.00, 0.10, 0.01],
    "Mean F1 Score": [0.811411, 0.795743, 0.785356, 0.777190, 0.755107, 0.504955],
    "Ranking": [1, 2, 3, 4, 5, 6],
})
SVM_BEST_PARAM = "C = 1"
SVM_TRAIN_TIME = 3.0375

NB_CONF_MATRIX = np.array([[2302, 206], [571, 689]])
SVM_CONF_MATRIX = np.array([[2317, 191], [237, 1023]])

FILTERED_EXAMPLES = {
    "Bukan Bahasa Indonesia": [
        "Corruption probe, BGN reshuffle test future of MBG program",
        "Indonesia has completed 222 Nutrition Fulfillment Service Units across 30 provinces to support President Prabowo's free meals program.",
    ],
    "Akun Media/News": [
        "Deretan Hoaks Penghentian Program MBG, Simak Faktanya #CekFakta",
        "Kepala Badan Gizi Nasional Nanik Sudaryanti Deyang menyebut Dewan Pengarah BGN akan diisi ahli gizi dan dokter anak.",
    ],
    "Gaya Penulisan Berita": [
        "Pelaksanaan Program Makan Bergizi Gratis (MBG) di SD YPPK St. Pontianus Ngondwe Ilwayab menjadi bukti nyata komitmen bersama dalam mendukung tumbuh kembang generasi penerus bangsa.",
        "Hasil survei Poltracking: Makan Bergizi Gratis (MBG) jadi program paling dirasakan manfaatnya rakyat (27,6% responden), di atas KIS, KIP & layanan kesehatan gratis.",
    ],
    "Template/Ajakan Generik": [
        "Dukung dan sukseskan program MBG",
        "Mari bersama-sama kita kawal dan sukseskan program makan bergizi gratis MBG yang kini tampil lebih berkualitas dan tepat sasaran untuk Indonesia.",
    ],
    "Meme/Noise": [
        "keren bgt pelatihnya ini, suruh mbg stop biar pelatih yg satu ini yg memasak",
        "Stop MBG !!! Biarkan abah yang memasakkk !!",
    ],
    "Hashtag Campaign": [
        "MBG tetap jalan, ekonomi tetap aman. Pernyataan bahwa S&P tidak meributkan MBG jadi sinyal positif. #MBGTakGangguAPBN #SPTidakMeributkanMBG",
       "Program MBG Papua Barat Daya Untuk Masa Depan Indonesia EMas 2045. #DukungMBG #PapuaBaratDaya https://t.co/vJv8gUCDdT",
    ],
    "Frasa Mobilisasi": [
        "Dalam rangka mendukung keberhasilan Program Makan Bergizi Gratis (MBG) serta memastikan makanan yang disajikan memenuhi standar kesehatan dan keamanan pangan.",
        "MBG bukan sekadar program makan gratis, tetapi investasi sumber daya manusia menuju Indonesia Emas 2045.",
    ],
}

# =====================================================================================
# LOADER (dengan cache + spinner)
# =====================================================================================

@st.cache_resource(show_spinner="Memuat model Naive Bayes & SVM...")
def load_models():
    paths = {
        "vectorizer": os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
        "nb": os.path.join(MODEL_DIR, "naive_bayes_model.pkl"),
        "svm": os.path.join(MODEL_DIR, "svm_model.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None, None
    vectorizer = joblib.load(paths["vectorizer"])
    nb_model = joblib.load(paths["nb"])
    svm_model = joblib.load(paths["svm"])
    return vectorizer, nb_model, svm_model


@st.cache_data(show_spinner="Memuat dataset...")
def load_csv_cached(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def generate_wordcloud_png(text_blob, colormap):
    """Generate wordcloud sebagai PNG bytes, di-cache berdasarkan isi teks +
    colormap. Tanpa cache ini, wordcloud (proses CPU-berat) dibuat ULANG dari
    nol setiap kali halaman Eksplorasi Data dirender ulang -- termasuk saat
    st.rerun() dipicu oleh klik navbar di halaman LAIN yang kebetulan membuat
    seluruh script jalan ulang. Ini salah satu penyebab utama respons lambat."""
    import io
    from wordcloud import WordCloud

    wc = WordCloud(width=500, height=350, background_color="white", colormap=colormap).generate(text_blob)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def run_prediction(text_for_vectorizer, vectorizer, nb_model, svm_model):
    X_input = vectorizer.transform([text_for_vectorizer])
    nb_pred_raw = nb_model.predict(X_input)[0]
    svm_pred_raw = svm_model.predict(X_input)[0]
    result = {
        "nb_pred": LABEL_MAP.get(nb_pred_raw, str(nb_pred_raw)),
        "svm_pred": LABEL_MAP.get(svm_pred_raw, str(svm_pred_raw)),
        "nb_proba": None,
        "svm_score": None,
    }
    if hasattr(nb_model, "predict_proba"):
        proba = nb_model.predict_proba(X_input)[0]
        result["nb_proba"] = pd.DataFrame({
            "Kelas": [LABEL_MAP.get(c, str(c)) for c in nb_model.classes_],
            "Probabilitas": proba,
        })
    if hasattr(svm_model, "decision_function"):
        result["svm_score"] = svm_model.decision_function(X_input)[0]
    return result


def show_prediction_result(result):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="card"><div class="chip chip-nb">NB</div>
        <h4>Naive Bayes</h4><h2 style="color:var(--text-color)">{result['nb_pred']}</h2></div>""",
        unsafe_allow_html=True)
        if result["nb_proba"] is not None:
            st.bar_chart(result["nb_proba"].set_index("Kelas"))
    with col2:
        st.markdown(f"""<div class="card"><div class="chip chip-svm">SVM</div>
        <h4>SVM</h4><h2 style="color:var(--text-color)">{result['svm_pred']}</h2></div>""",
        unsafe_allow_html=True)
        if result["svm_score"] is not None:
            st.metric("Skor Decision Function", f"{result['svm_score']:.3f}")
            st.caption(f"Skor < 0 → {LABEL_MAP.get(0)}, skor > 0 → {LABEL_MAP.get(1)}")

    st.write("")
    if result["nb_pred"] == result["svm_pred"]:
        st.success(f"Kedua model kompak: sentimennya **{result['nb_pred']}**", icon=":material/check_circle:")
    else:
        st.info(f"Dua model beda pendapat nih — Naive Bayes bilang **{result['nb_pred']}**, SVM bilang **{result['svm_pred']}**", icon=":material/info:")


# =====================================================================================
# TOPBAR / NAVBAR (pengganti sidebar) — gaya landing page: logo kiri, menu kanan,
# CTA "Try Now" nunjuk ke halaman prediksi
# =====================================================================================

# (page_id, icon, label_singkat) — page_id dipakai buat routing (jangan diubah),
# label_singkat yang tampil di tombol supaya nggak wrap 2 baris
NAV_ITEMS = [
    ("Beranda", "home", "Beranda"),
    ("Eksplorasi Data", "bar_chart", "Eksplorasi"),
    ("Cara Kerja Model", "psychology", "Cara Kerja"),
    ("Evaluasi Model", "insights", "Evaluasi"),
]
CTA_PAGE_ID = "Uji Coba Prediksi"

if "current_page" not in st.session_state:
    st.session_state.current_page = NAV_ITEMS[0][0]

with st.container(key="navbar"):
    brand_col, *nav_cols, cta_col = st.columns(
        [2.1] + [0.85] * len(NAV_ITEMS) + [1.1]
    )

    with brand_col:
        st.markdown(
            """<div class="navbar-brand">
                <span class="material-symbols-outlined">nutrition</span>
                <div>Sentimen MBG<div class="navbar-tagline">Naive Bayes vs SVM</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

    for col, (page_id, icon, label) in zip(nav_cols, NAV_ITEMS):
        is_active = st.session_state.current_page == page_id
        with col:
            if st.button(
                label,
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                icon=f":material/{icon}:",
            ):
                st.session_state.current_page = page_id
                st.rerun()

    with cta_col:
        with st.container(key="navcta"):
            if st.button(
                "Try Now",
                key="nav_cta_btn",
                use_container_width=True,
                type="primary",
                icon=":material/bolt:",
            ):
                st.session_state.current_page = CTA_PAGE_ID
                st.rerun()

    page = st.session_state.current_page

# =====================================================================================
# HALAMAN 1 - BERANDA
# =====================================================================================

if page == "Beranda":
    # ---------------------------------------------------------------------
    # HERO — dua kolom: teks + CTA di kiri, kartu "Live Preview" di kanan
    # ---------------------------------------------------------------------
    hero_l, hero_r = st.columns([1.15, 1], gap="large")
    with hero_l:
        st.markdown(
            f"""
            <div class="reveal-left">
                <span class="pill-badge">
                    <span class="material-symbols-outlined" style="font-size:0.9rem;vertical-align:-2px;">bolt</span>
                    Didukung Machine Learning
                </span>
                <div class="hero-split">
                    <h1>Analisis Sentimen <span class="accent">Program Makan<br>Bergizi Gratis</span></h1>
                    <p class="hero-desc">Melihat bagaimana warganet merespons Program Makan Bergizi Gratis
                    (MBG) di X (Twitter), dianalisis menggunakan dua algoritma machine learning:
                    Naive Bayes dan Support Vector Machine.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bcol1, bcol2 = st.columns([1, 1.6])
        with bcol1:
            if st.button("Coba Sekarang", key="hero_cta_btn", type="primary",
                         icon=":material/bolt:", use_container_width=True):
                st.session_state.current_page = "Uji Coba Prediksi"
                st.rerun()
        st.markdown(
            f"""
            <div class="reveal-left reveal-d2 hero-stats">
                <div><div class="stat-num">30.692</div><div class="stat-label">Dataset/Tweet</div></div>
                <div><div class="stat-num">  2  </div><div class="stat-label">ML Models</div></div>
                <div><div class="stat-num">{SVM_METRICS['Accuracy']}%</div><div class="stat-label">Akurasi Terbaik(SVM)</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_r:
        st.markdown(
            f"""
            <div class="reveal-right preview-card">
                <div class="preview-title">
                    <span class="material-symbols-outlined" style="font-size:1rem;vertical-align:-2px;">monitoring</span>
                    Sentiment Preview
                </div>
                <div class="preview-row">
                    <span class="preview-text">"program MBG itu kalau didesign dengan benar dan sesuai niat presiden... itu sangat bagus lho. "</span>
                    <span class="sent-badge sent-badge-pos">Positif</span>
                </div>
                <div class="preview-row">
                    <span class="preview-text">"Stop Mbg, ladang koruptor ngabisin uang anggaran aja."</span>
                    <span class="sent-badge sent-badge-neg">Negatif</span>
                </div>
                <div class="preview-row">
                    <span class="preview-text">"mbg dibubarin aja please maksa banget, kita rakyat Indonesia janji deh engga akan mengolok ngolok program gagal santuy kalo bubar malah pada seneng kok."</span>
                    <span class="sent-badge sent-badge-neg">Negatif</span>
                </div>
                <div class="preview-row">
                    <span class="preview-text">"Anak-anak jadi lebih semangat sekolah."</span>
                    <span class="sent-badge sent-badge-pos">Positif</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # ---------------------------------------------------------------------
    # ALUR PENELITIAN — numbered flow dengan panah penghubung
    # ---------------------------------------------------------------------
    st.markdown('<div class="section-title reveal-up">Alur Penelitian</div>', unsafe_allow_html=True)
    steps = [
        ("Scraping", "Ambil data tweet soal MBG lewat Apify"),
        ("Filtering", "Buang duplikat, akun media, meme, campaign terorganisir"),
        ("Preprocessing", "Cleaning, normalisasi, stemming, & pemeriksaan kualitas data"),
        ("Labeling", "Kasih label sentimen pakai pendekatan lexicon-based"),
        ("Modeling", "Latih & tuning Naive Bayes dan SVM dengan TF-IDF"),
        ("Evaluasi", "Bandingkan performa dua model pakai data testing"),
    ]
    flow_html = '<div class="reveal-up flow-row">'
    for i, (title, desc) in enumerate(steps, start=1):
        flow_html += f"""<div class="flow-item"><div class="step-num">{i}</div>
            <h4>{title}</h4><p>{desc}</p></div>"""
        if i < len(steps):
            flow_html += '<span class="material-symbols-outlined flow-arrow">arrow_forward</span>'
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ---------------------------------------------------------------------
    # MENGAPA — feature rows (kiri) + kartu statistik sebaran sentimen (kanan)
    # ---------------------------------------------------------------------
    st.markdown('<div class="section-title reveal-up">Mengapa Sentiment Analisys?</div>', unsafe_allow_html=True)
    why_l, why_r = st.columns([1.05, 1], gap="large")
    FEATURES = [
        ("bolt", "Prediksi Real-time", "Model langsung menebak sentimen sebuah opini begitu teksnya dimasukkan, tanpa perlu training ulang."),
        ("translate", "Bahasa Indonesia", "Preprocessing dirancang khusus untuk teks Bahasa Indonesia, termasuk stemming Sastrawi dan stopword removal."),
        ("verified", "Akurat & Teruji", "Kedua model divalidasi lewat hyperparameter tuning dan diuji pada ribuan data tweet berlabel."),
        ("touch_app", "Mudah Digunakan", "Tinggal tulis opini sendiri atau pilih dari data yang sudah berlabel, hasilnya langsung terlihat."),
    ]
    with why_l:
        feat_html = '<div class="reveal-left">'
        for icon, title, desc in FEATURES:
            feat_html += f"""<div class="feature-row">
                <div class="feature-icon-box"><span class="material-symbols-outlined">{icon}</span></div>
                <div><h4>{title}</h4><p>{desc}</p></div>
            </div>"""
        feat_html += "</div>"
        st.markdown(feat_html, unsafe_allow_html=True)
    with why_r:
        dist_df_home = pd.DataFrame(SENTIMENT_DIST)
        bars_html = ""
        for _, row in dist_df_home.iterrows():
            color = COLORS.get(row["Sentimen"].lower(), COLORS["dark"])
            bars_html += f"""<div class="stat-bar-label"><span>{row['Sentimen']}</span><span>{row['Persentase']}%</span></div>
            <div class="stat-bar-track"><div class="stat-bar-fill" style="width:{row['Persentase']}%;background:{color};"></div></div>"""
        metrics_html = ""
        for label, color in [("Accuracy", COLORS["primary"]), ("Precision", COLORS["secondary"]),
                              ("Recall", COLORS["negatif"]), ("F1-Score", COLORS["positif"])]:
            metrics_html += f"""<div class="mini-metric" style="background:{color}44;">
                <div class="num">{SVM_METRICS[label]}%</div><div class="label">{label} (SVM)</div></div>"""
        st.markdown(
            f"""
            <div class="reveal-right preview-card">
                <div class="stat-card-title">Sebaran Sentimen Dataset</div>
                {bars_html}
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:1rem;">
                    {metrics_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # ---------------------------------------------------------------------
    # TEKNOLOGI YANG DIGUNAKAN
    # ---------------------------------------------------------------------
    st.markdown('<div class="section-title reveal-up">Teknologi yang Digunakan</div>', unsafe_allow_html=True)
    TECH_STACK = [
        ("code", "Python", "Bahasa utama untuk ML & pemrosesan data"),
        ("hub", "Scikit-learn", "Library ML untuk Implementasi Naive Bayes dan SVM"),
        ("translate", "NLP Tools", "NLTK dan Sastrawi untuk Preprocessing teks Bahasa Indonesia"),
        ("dashboard", "Streamlit", "Dashboard interaktif"),
    ]
    tcols = st.columns(4)
    for i, (col, (icon, title, desc)) in enumerate(zip(tcols, TECH_STACK)):
        with col:
            st.markdown(
                f"""<div class="reveal-up reveal-d{i+1} tech-item">
                    <div class="tech-icon-box"><span class="material-symbols-outlined">{icon}</span></div>
                    <h4>{title}</h4><p>{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    mcol1, mcol2 = st.columns(2, gap="large")
    with mcol1:
        st.markdown(
            f"""
            <div class="reveal-left card">
                <span class="material-symbols-outlined" style="font-size:1.8rem;color:{COLORS['primary']}">speed</span>
                <h4>Naive Bayes</h4>
                <p class="muted" style="font-size:0.85rem;">Algoritma probabilistik berbasis Teorema Bayes yang efisien
                untuk klasifikasi teks dengan asumsi independensi antar fitur.</p>
                <div class="check-item"><span class="material-symbols-outlined">check_circle</span> Training sangat cepat</div>
                <div class="check-item"><span class="material-symbols-outlined">check_circle</span> Efisien untuk dataset besar</div>
                <div class="check-item"><span class="material-symbols-outlined">check_circle</span> Cocok untuk text classification</div>
                <div class="accuracy-banner" style="background:{COLORS['primary']}55;">
                    <div class="num">{NB_METRICS['Accuracy']}%</div><div class="label">Accuracy</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            f"""
            <div class="reveal-right card">
                <span class="material-symbols-outlined" style="font-size:1.8rem;color:{COLORS['primary']}">hub</span>
                <h4>Support Vector Machine</h4>
                <p class="muted" style="font-size:0.85rem;">Algoritma supervised learning yang mencari garis pemisah
                (hyperplane) optimal antar kelas dengan margin maksimal.</p>
                <div class="check-item"><span class="material-symbols-outlined">check_circle</span> Akurasi lebih tinggi</div>
                <div class="check-item"><span class="material-symbols-outlined">check_circle</span> Robust terhadap overfitting</div>
                <div class="check-item"><span class="material-symbols-outlined">check_circle</span> Efektif untuk data dimensi tinggi</div>
                <div class="accuracy-banner" style="background:{COLORS['secondary']}77;">
                    <div class="num">{SVM_METRICS['Accuracy']}%</div><div class="label">Best Accuracy</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # ---------------------------------------------------------------------
    # CTA PENUTUP
    # ---------------------------------------------------------------------
    st.markdown(
        """
        <div class="reveal-up cta-banner">
            <h3>Coba Analisis Sentimen Sekarang</h3>
            <p>Tulis opini bebas soal MBG, atau coba langsung ke data yang sudah berlabel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cta_l, cta_mid, cta_r = st.columns([1, 0.6, 1])
    with cta_mid:
        with st.container(key="cta_banner_btn"):
            if st.button("Analisis Sentimen", key="home_bottom_cta", icon=":material/bolt:", use_container_width=True):
                st.session_state.current_page = "Uji Coba Prediksi"
                st.rerun()

# =====================================================================================
# HALAMAN 2 - EKSPLORASI DATA
# =====================================================================================

elif page == "Eksplorasi Data":
    st.markdown('<div class="section-title reveal-up">Eksplorasi Data</div>', unsafe_allow_html=True)
    st.caption(
        "Pengurangan jumlah data terjadi dalam **2 tahap yang berbeda**: (1) filtering data mentah "
        "hasil scraping, dan (2) pemeriksaan kualitas data setelah text preprocessing."
    )

    # -----------------------------------------------------------------
    # TAHAP 1: FILTERING DATA MENTAH (scraping -> 21.231)
    # -----------------------------------------------------------------
    st.markdown("**Tahap 1: Filtering Data Mentah (Menghapus Data Tidak Relevan)**")
    st.caption(
        "Menyaring tweet hasil scraping agar hanya berisi opini asli warganet tentang MBG"
        "menghapus duplikat, tweet non-Indonesia, akun media/berita, template campaign, "
        "meme, hashtag campaign, dan frasa mobilisasi."
    )
    filtering_df = pd.DataFrame(FILTERING_FUNNEL)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = sns.barplot(data=filtering_df, y="Tahap", x="Jumlah Data", hue="Tahap",
                        palette=sns.color_palette("Blues_r", len(filtering_df)), legend=False, ax=ax)
    for i, v in enumerate(filtering_df["Jumlah Data"]):
        ax.text(v + 250, i, f"{v:,}", va="center", fontsize=9, color="#2D3436")
    ax.set_xlabel("Jumlah Data")
    ax.set_ylabel("")
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.metric("Data Awal (Scraping)", f"{FILTERING_FUNNEL['Jumlah Data'][0]:,}")
    with f2:
        st.metric("Setelah Filtering Tahap 1", f"{FILTERING_FUNNEL['Jumlah Data'][-1]:,}")
    with f3:
        removed1 = FILTERING_FUNNEL["Jumlah Data"][0] - FILTERING_FUNNEL["Jumlah Data"][-1]
        st.metric("Data Dihapus", f"{removed1:,}", delta=f"-{removed1/FILTERING_FUNNEL['Jumlah Data'][0]*100:.1f}%")

    st.write("")

    # -----------------------------------------------------------------
    # TAHAP 2: PEMERIKSAAN KUALITAS PASCA-PREPROCESSING (21.231 -> 19.585)
    # -----------------------------------------------------------------
    st.markdown('<div class="section-title reveal-up">Tahap 2: Pemeriksaan Kualitas Data Pasca-Preprocessing</div>', unsafe_allow_html=True)
    st.caption(
        "Dilakukan **setelah** text preprocessing (cleaning, case folding, normalisasi, "
        "stopword removal, stemming). Bukan bagian dari filtering tahap 1. Untuk memastikan "
        "hanya data berkualitas yang lanjut ke tahap pelabelan sentimen."
    )

    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.metric("Sebelum Pemeriksaan", f"{QUALITY_CHECK['before']:,}")
    with qc2:
        st.metric("Setelah Pemeriksaan", f"{QUALITY_CHECK['after']:,}")
    with qc3:
        removed2 = QUALITY_CHECK["before"] - QUALITY_CHECK["after"]
        st.metric("Data Dihapus", f"{removed2:,}", delta=f"-{removed2/QUALITY_CHECK['before']*100:.1f}%")

    qc_col_chart, qc_col_list = st.columns([1, 1.3], gap="large")
    with qc_col_chart:
        qc_bar_df = pd.DataFrame({
            "Tahap": ["Sebelum Pemeriksaan", "Setelah Pemeriksaan"],
            "Jumlah Data": [QUALITY_CHECK["before"], QUALITY_CHECK["after"]],
        })
        st.bar_chart(qc_bar_df.set_index("Tahap"), color=COLORS["secondary"])
    with qc_col_list:
        st.markdown("**Rincian pemeriksaan yang dilakukan:**")
        for step in QUALITY_CHECK["steps"]:
            st.markdown(
                f"""<div class="card" style="border-left:4px solid {COLORS['secondary']};margin-bottom:0.5rem;padding:0.7rem 1rem;">{step}</div>""",
                unsafe_allow_html=True,
            )

    st.write("")

    # -----------------------------------------------------------------
    # SEBARAN SENTIMEN
    # -----------------------------------------------------------------
    st.markdown('<div class="section-title reveal-up">Sebaran Sentimen (Hasil Lexicon-Based Labeling)</div>', unsafe_allow_html=True)
    dist_df = pd.DataFrame(SENTIMENT_DIST)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        colors = [COLORS["negatif"], COLORS["positif"], COLORS["netral"]]
        wedges, texts, autotexts = ax2.pie(
            dist_df["Jumlah"], labels=dist_df["Sentimen"], autopct="%1.2f%%",
            colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_color("#2D3436")
            t.set_fontweight("bold")
        ax2.axis("equal")
        st.pyplot(fig2)
        plt.close(fig2)
    with c2:
        for _, row in dist_df.iterrows():
            color = COLORS.get(row["Sentimen"].lower(), COLORS["dark"])
            st.markdown(f"""
            <div class="card" style="margin-bottom:0.7rem; border-left:5px solid {color};">
                <b>{row['Sentimen']}</b> — {row['Jumlah']:,} data ({row['Persentase']}%)
            </div>""", unsafe_allow_html=True)
        st.info(
            "Sentimen Netral dieliminasi dari proses training untuk menghindari masalah class imbalance. "
            "Model Naive Bayes dan SVM dikembangkan menggunakan pendekatan Klasifikasi Biner (Positif vs Negatif) guna mengoptimalkan akurasi prediksi pada polaritas sentimen utama.",
            icon=":material/info:",
        )

    st.markdown('<div class="section-title reveal-up">Split Data Training vs Testing</div>', unsafe_allow_html=True)
    split_df = pd.DataFrame({"Set": list(SPLIT_DATA.keys()), "Jumlah Data": list(SPLIT_DATA.values())})
    st.bar_chart(split_df.set_index("Set"), color=COLORS["primary"])

    st.markdown('<div class="section-title reveal-up">Contoh Data yang Dibuang Saat Cleaning</div>', unsafe_allow_html=True)
    st.caption("Data-data ini nggak dipakai karena bukan cerminan opini asli warganet soal MBG.")
    filter_tabs = st.tabs(list(FILTERED_EXAMPLES.keys()))
    for tab, (reason, examples) in zip(filter_tabs, FILTERED_EXAMPLES.items()):
        with tab:
            for ex in examples:
                st.markdown(f"""<div class="card" style="border-left:4px solid #94A3B8;">{ex}</div>""",
                            unsafe_allow_html=True)
                st.write("")

    st.markdown('<div class="section-title reveal-up">Wordcloud Negatif vs Positif</div>', unsafe_allow_html=True)
    df_wc_source = load_csv_cached(DATA_LABELED_3CLASS_PATH)
    if df_wc_source is not None:
        df_wc = df_wc_source[df_wc_source["label"].isin(["negatif", "positif"])]
        try:
            from wordcloud import WordCloud
            with st.spinner("Membuat wordcloud..."):
                col_neg, col_pos = st.columns(2)
                for col, label, cmap in [(col_neg, "negatif", "Reds"), (col_pos, "positif", "Greens")]:
                    text_blob = " ".join(df_wc[df_wc["label"] == label]["hasil_preprocessing"].astype(str))
                    with col:
                        st.markdown(f"**{label.capitalize()}**")
                        if text_blob.strip():
                            png_bytes = generate_wordcloud_png(text_blob, cmap)
                            st.image(png_bytes, use_container_width=True)
        except ImportError:
            st.error("Package 'wordcloud' belum terpasang. Tambahkan ke requirements.txt", icon=":material/error:")
    else:
        st.warning(f"File `{DATA_LABELED_3CLASS_PATH}` tidak ditemukan.", icon=":material/warning:")

# =====================================================================================
# HALAMAN 3 - CARA KERJA MODEL
# =====================================================================================

elif page == "Cara Kerja Model":
    st.markdown('<div class="section-title reveal-up">Cara Kerja Model</div>', unsafe_allow_html=True)

    st.markdown(
        "Sebelum masuk rumus-rumus, ini gambaran umum cara kerja sistemnya, "
        "supaya siapa pun yang baca — nggak harus paham matematika — tetap ngerti intinya."
    )

    st.markdown(f"""<div class="reveal-left card"><span class="material-symbols-outlined" style="font-size:1.8rem;color:{COLORS['primary']}">calculate</span>
    <h4>1. TF-IDF — Mengubah Teks Jadi Angka</h4>
    <p>Komputer nggak bisa langsung "membaca" kalimat kayak manusia, jadi teks harus diubah
    jadi angka dulu. TF-IDF adalah cara memberi <b>bobot kepentingan</b> ke tiap kata dalam sebuah tweet.</p>
    <p>Logikanya sederhana: kalau sebuah kata sering muncul di satu tweet tapi jarang muncul
    di tweet-tweet lain, kata itu dianggap ciri khas tweet tersebut dan diberi bobot tinggi.
    Sebaliknya, kata umum yang muncul di hampir semua tweet (misalnya "program", "mbg", "yang")
    dianggap kurang informatif meskipun sering muncul, jadi bobotnya rendah.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("Lihat rumus formal TF-IDF", icon=":material/function:"):
        st.latex(r"TF(t, d) = \frac{\text{jumlah kemunculan kata } t \text{ di dokumen } d}{\text{jumlah total kata pada dokumen } d}")
        st.latex(r"IDF(t) = \log \left( \frac{N}{df(t)} \right) + 1")
        st.latex(r"TFIDF(t, d) = TF(t, d) \times IDF(t)")
        st.caption("N = jumlah total dokumen, df(t) = jumlah dokumen yang mengandung kata t")

    with st.expander("Coba hitung TF-IDF sendiri", icon=":material/search:"):
        default_docs = "makan bergizi gratis membantu anak sekolah\nprogram makan bergizi gratis kualitasnya buruk\nsaya setuju dengan program makan gratis ini"
        docs_input = st.text_area("Masukkan beberapa kalimat (satu kalimat per baris)", value=default_docs, height=110)
        docs = [d for d in docs_input.split("\n") if d.strip()]
        if len(docs) >= 2:
            demo_vec = TfidfVectorizer()
            tfidf_matrix = demo_vec.fit_transform(docs)
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=demo_vec.get_feature_names_out()).round(3)
            st.dataframe(tfidf_df, use_container_width=True)
        else:
            st.info("Masukkan minimal 2 kalimat dulu ya.", icon=":material/info:")

    st.markdown(f"""<div class="reveal-right card"><span class="material-symbols-outlined" style="font-size:1.8rem;color:{COLORS['primary']}">mail</span>
    <h4>2. Naive Bayes — Mirip Cara Kerja Filter Spam</h4>
    <p>Naive Bayes menebak sentimen berdasarkan kata-kata yang muncul, mirip cara email
    mendeteksi spam. Model belajar dari data training: kata apa saja yang sering muncul di
    tweet negatif, dan kata apa yang sering muncul di tweet positif.</p>
    <p>Saat ada tweet baru, model menghitung: "kata-kata di tweet ini lebih mirip pola negatif
    atau pola positif?" lalu memilih yang peluangnya paling besar. Disebut "naive" (naif) karena
    model ini menganggap tiap kata muncul saling independen — padahal kenyataannya kata-kata
    dalam kalimat saling berhubungan. Meski begitu, asumsi sederhana ini biasanya tetap cukup akurat untuk klasifikasi teks.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("Lihat rumus formal Naive Bayes", icon=":material/function:"):
        st.latex(r"P(y \mid x_1, ..., x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, ..., x_n)}")
        st.markdown("Karena penyebutnya sama untuk semua kelas, cukup cari kelas yang memaksimalkan pembilangnya:")
        st.latex(r"\hat{y} = \arg\max_{y} \; P(y) \prod_{i=1}^{n} P(x_i \mid y)")
        st.markdown("Untuk data teks, dipakai Laplace Smoothing (alpha) supaya kata yang belum pernah muncul tidak membuat probabilitas jadi nol:")
        st.latex(r"P(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}")
        st.caption(f"Parameter terbaik hasil tuning pada penelitian ini: **{NB_BEST_PARAM}**")

    st.markdown(f"""<div class="reveal-left card"><span class="material-symbols-outlined" style="font-size:1.8rem;color:{COLORS['primary']}">straighten</span>
    <h4>3. SVM — Menarik Garis Pemisah Terbaik</h4>
    <p>Bayangkan tiap tweet sebagai satu titik di sebuah bidang. SVM (Support Vector Machine)
    mencari garis pemisah terbaik antara kumpulan titik "negatif" dan kumpulan titik "positif".</p>
    <p>Bukan sembarang garis — SVM mencari garis dengan jarak (margin) paling lebar ke titik-titik
    terdekat dari kedua sisi. Semakin lebar marginnya, semakin percaya diri model saat menghadapi
    data baru. Titik-titik yang posisinya paling dekat dengan garis pemisah itu disebut
    "support vector", karena merekalah yang menentukan letak garis — dari situ nama SVM berasal.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("Lihat rumus formal SVM", icon=":material/function:"):
        st.latex(r"f(x) = w \cdot x + b")
        st.markdown("Fungsi keputusan klasifikasi:")
        st.latex(r"\hat{y} = \text{sign}(w \cdot x + b)")
        st.markdown("Parameter (w, b) dicari dengan meminimalkan fungsi berikut (soft-margin SVM):")
        st.latex(r"\min_{w, b} \; \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{m} \max(0, 1 - y_i(w \cdot x_i + b))")
        st.caption(
            f"C adalah parameter yang mengatur trade-off antara margin lebar dan kesalahan klasifikasi. "
            f"Parameter terbaik hasil tuning: **{SVM_BEST_PARAM}**"
        )

# =====================================================================================
# HALAMAN 4 - EVALUASI MODEL
# =====================================================================================

elif page == "Evaluasi Model":
    st.markdown('<div class="section-title reveal-up">Evaluasi & Perbandingan Model</div>', unsafe_allow_html=True)

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown(f"**Naive Bayes** — parameter terbaik `{NB_BEST_PARAM}` · waktu training {NB_TRAIN_TIME}s")
        st.dataframe(NB_TUNING, hide_index=True, use_container_width=True)
    with tcol2:
        st.markdown(f"**SVM** — parameter terbaik `{SVM_BEST_PARAM}` · waktu training {SVM_TRAIN_TIME}s")
        st.dataframe(SVM_TUNING, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-title reveal-up">Perbandingan Metrik</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    for col, m in zip(metric_cols, NB_METRICS.keys()):
        with col:
            st.markdown(f"""<div class="card">
                <p style="margin:0;font-size:0.85rem" class="muted">{m}</p>
                <p style="margin:0.3rem 0 0 0;font-size:1.05rem">
                    <span style="background:{COLORS['negatif']}66;color:var(--text-color);padding:3px 10px;border-radius:8px;font-weight:700;">NB {NB_METRICS[m]}%</span>
                </p>
                <p style="margin:0.3rem 0 0 0;font-size:1.05rem">
                    <span style="background:{COLORS['positif']}66;color:var(--text-color);padding:3px 10px;border-radius:8px;font-weight:700;">SVM {SVM_METRICS[m]}%</span>
                </p>
            </div>""", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "Metrik": list(NB_METRICS.keys()),
        "Naive Bayes (%)": list(NB_METRICS.values()),
        "SVM (%)": list(SVM_METRICS.values()),
    })
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_df["Metrik"]))
    width = 0.35
    ax4.bar(x - width/2, metrics_df["Naive Bayes (%)"], width, label="Naive Bayes", color=COLORS["dark"])
    ax4.bar(x + width/2, metrics_df["SVM (%)"], width, label="SVM", color=COLORS["primary"])
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_df["Metrik"])
    ax4.set_ylabel("Persentase (%)")
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.legend()
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown('<div class="section-title reveal-up">Confusion Matrix</div>', unsafe_allow_html=True)
    st.caption(
        "Confusion matrix membandingkan label **Aktual** (baris) dengan label hasil "
        "**Prediksi** model (kolom). Kalau belum familiar cara bacanya, lihat penjelasan "
        "& diagram batang di bawah tiap matrix."
    )

    def _conf_explanation_and_chart(cm, model_name):
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        total = tn + fp + fn + tp
        benar = tn + tp
        salah = fp + fn

        st.markdown(
            f"""
            <div class="card" style="margin-top:0.6rem;">
                <p style="margin:0 0 0.4rem 0;">
                    <span class="chip" style="background:{COLORS['secondary']}">Akurat</span>
                    Benar menebak <b>{benar:,}</b> dari <b>{total:,}</b> data uji
                    (<b>{benar/total*100:.1f}%</b>)
                </p>
                <p style="margin:0;" class="muted">
                    • Aktual Negatif ditebak Negatif (benar): <b>{tn:,}</b><br>
                    • Aktual Positif ditebak Positif (benar): <b>{tp:,}</b><br>
                    • Aktual Negatif tapi ditebak Positif (salah): <b>{fp:,}</b><br>
                    • Aktual Positif tapi ditebak Negatif (salah): <b>{fn:,}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bar chart: jumlah Aktual vs Diprediksi per kelas — lebih mudah dibaca
        # orang awam dibanding angka mentah di confusion matrix.
        aktual_neg, aktual_pos = tn + fp, fn + tp
        pred_neg, pred_pos = tn + fn, fp + tp

        chart_df = pd.DataFrame({
            "Negatif": [aktual_neg, pred_neg],
            "Positif": [aktual_pos, pred_pos],
        }, index=["Aktual", "Diprediksi"])

        figc, axc = plt.subplots(figsize=(4, 3))
        x = np.arange(2)
        width = 0.35
        axc.bar(x - width/2, chart_df["Negatif"], width, label="Negatif", color=COLORS["primary"])
        axc.bar(x + width/2, chart_df["Positif"], width, label="Positif", color=COLORS["secondary"])
        axc.set_xticks(x)
        axc.set_xticklabels(chart_df.index)
        axc.set_ylabel("Jumlah Data")
        axc.spines[['top', 'right']].set_visible(False)
        axc.legend(fontsize=8)
        figc.tight_layout()
        st.pyplot(figc)
        plt.close(figc)

    ccol1, ccol2 = st.columns(2)
    labels = ["Negatif", "Positif"]
    with ccol1:
        st.markdown("**Naive Bayes**")
        fig5, ax5 = plt.subplots(figsize=(4, 4))
        sns.heatmap(NB_CONF_MATRIX, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax5, cbar=False)
        ax5.set_xlabel("Prediksi")
        ax5.set_ylabel("Aktual")
        st.pyplot(fig5)
        plt.close(fig5)
        _conf_explanation_and_chart(NB_CONF_MATRIX, "Naive Bayes")
    with ccol2:
        st.markdown("**SVM**")
        fig6, ax6 = plt.subplots(figsize=(4, 4))
        sns.heatmap(SVM_CONF_MATRIX, annot=True, fmt="d", cmap="BuGn",
                    xticklabels=labels, yticklabels=labels, ax=ax6, cbar=False)
        ax6.set_xlabel("Prediksi")
        ax6.set_ylabel("Aktual")
        st.pyplot(fig6)
        plt.close(fig6)
        _conf_explanation_and_chart(SVM_CONF_MATRIX, "SVM")

    st.success(
        f"SVM (akurasi {SVM_METRICS['Accuracy']}%, F1 {SVM_METRICS['F1-Score']}%) unggul dari "
        f"Naive Bayes (akurasi {NB_METRICS['Accuracy']}%, F1 {NB_METRICS['F1-Score']}%) di semua metrik.",
        icon=":material/check_circle:",
    )

# =====================================================================================
# HALAMAN 5 - UJI COBA PREDIKSI
# =====================================================================================

elif page == "Uji Coba Prediksi":
    st.markdown('<div class="section-title reveal-up">Coba Analisis Sentimen</div>', unsafe_allow_html=True)

    vectorizer, nb_model, svm_model = load_models()
    models_ready = vectorizer is not None

    if not models_ready:
        st.error(
            "Model belum ditemukan. Taruh 3 file berikut di folder `models/`: "
            "`tfidf_vectorizer.pkl`, `naive_bayes_model.pkl`, `svm_model.pkl`",
            icon=":material/error:",
        )

    tab1, tab2 = st.tabs([":material/edit_note: Input Teks Baru", ":material/fact_check: Data Berlabel (Ground Truth)"])

    # ------------------------------------------------------------------------------
    # TAB 1: Input teks baru
    # ------------------------------------------------------------------------------
    with tab1:
        st.caption("Tulis opini bebas soal MBG, model akan menebak sentimennya.")
        input_text = st.text_area(
            "Teks",
            placeholder="Contoh: Program makan bergizi gratis ini sangat membantu anak-anak sekolah ",
            height=120,
            label_visibility="collapsed",
        )
        show_preprocess = st.checkbox("Tampilkan tahapan preprocessing", value=True)

        if st.button("Analisis Sentimen", type="primary", key="predict_btn_tab1", icon=":material/bolt:"):
            if not models_ready:
                st.error("Model belum siap.", icon=":material/error:")
            elif not PREPROCESSING_AVAILABLE:
                st.error(
                    "File `preprocessing.py` gagal di-import "
                    f"(error: `{PREPROCESSING_IMPORT_ERROR}`). Pastikan nltk & Sastrawi "
                    "sudah terpasang, dan `preprocessing.py` satu folder dengan `app.py`.",
                    icon=":material/error:",
                )
            elif not input_text.strip():
                st.warning("Tulis teksnya dulu ya.", icon=":material/warning:")
            else:
                with st.spinner("Memproses teks & menjalankan model..."):
                    steps = preprocess_with_steps(input_text)
                    cleaned = steps["final_text"]

                if show_preprocess:
                    st.markdown("**Tahapan Preprocessing**")
                    step_df = pd.DataFrame({
                        "Tahap": ["Cleaning", "Case Folding", "Tokenizing", "Normalization", "Stopword Removal", "Stemming"],
                        "Hasil": [
                            steps["cleaning"], steps["case_folding"], str(steps["tokenizing"]),
                            str(steps["normalization"]), str(steps["stopword_removal"]), str(steps["stemming"]),
                        ],
                    })
                    st.dataframe(step_df, hide_index=True, use_container_width=True)
                    st.code(cleaned if cleaned else "(kosong setelah preprocessing)")

                if not cleaned.strip():
                    st.warning("Teksnya jadi kosong setelah preprocessing (mungkin cuma berisi stopword). Coba kalimat lain.", icon=":material/warning:")
                else:
                    result = run_prediction(cleaned, vectorizer, nb_model, svm_model)
                    show_prediction_result(result)

    # ------------------------------------------------------------------------------
    # TAB 2: Data berlabel, pilih dari tabel
    # ------------------------------------------------------------------------------
    with tab2:
        st.caption(
            "Pilih salah satu tweet dari data yang sudah dilabeli, lalu cek apakah prediksi "
            "model cocok dengan label aslinya."
        )

        df_labeled = load_csv_cached(DATA_LABELED_BINARY_PATH)
        if df_labeled is None:
            st.warning(f"File `{DATA_LABELED_BINARY_PATH}` tidak ditemukan.", icon=":material/warning:")
        elif not models_ready:
            st.info("Menunggu model dimuat.", icon=":material/hourglass_empty:")
        else:
            filter_label = st.selectbox("Filter label asli", ["Semua", "Negatif", "Positif"])
            pool = df_labeled if filter_label == "Semua" else df_labeled[df_labeled["label"] == filter_label.lower()]
            pool = pool.reset_index(drop=True)

            n_show = st.slider("Jumlah data ditampilkan di tabel", 50, 200, 100, step=50)
            display_df = pool[["text", "label", "score"]].head(n_show).copy()
            display_df["label"] = display_df["label"].str.capitalize()

            st.caption(f"Menampilkan {len(display_df)} dari {len(pool):,} data. Klik satu baris di tabel untuk memilih.")

            selected_idx = None
            try:
                event = st.dataframe(
                    display_df, use_container_width=True, hide_index=True, height=320,
                    on_select="rerun", selection_mode="single-row", key="labeled_table",
                )
                if event and event.selection and event.selection.rows:
                    selected_idx = event.selection.rows[0]
            except TypeError:
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=320)
                selected_idx = st.selectbox(
                    "Streamlit versi ini belum mendukung klik tabel, pilih index manual:",
                    options=list(range(len(display_df))),
                )

            colA, colB = st.columns([1, 3])
            with colA:
                if st.button("Pilih Acak", icon=":material/casino:"):
                    selected_idx = random.randint(0, len(display_df) - 1)
                    st.session_state.labeled_selected_idx = selected_idx

            if selected_idx is not None:
                st.session_state.labeled_selected_idx = selected_idx

            if "labeled_selected_idx" in st.session_state and st.session_state.labeled_selected_idx < len(pool):
                row = pool.iloc[st.session_state.labeled_selected_idx]

                st.markdown("**Teks Terpilih:**")
                st.markdown(f"""<div class="card" style="border-left:4px solid {COLORS['primary']}">{row['text']}</div>""",
                            unsafe_allow_html=True)

                with st.expander("Lihat tahapan preprocessing (sudah dihitung sebelumnya)"):
                    step_df = pd.DataFrame({
                        "Tahap": ["Cleaning", "Case Folding", "Remove Char", "Tokenizing", "Normalization", "Stopword Removal", "Stemming", "Hasil Akhir"],
                        "Hasil": [
                            row["cleaning"], row["case_folding"], row["remove_char"], row["tokenizing"],
                            row["normalisasi"], row["stopword"], row["stemming"], row["hasil_preprocessing"],
                        ],
                    })
                    st.dataframe(step_df, hide_index=True, use_container_width=True)

                true_label = LABEL_MAP.get(int(row["label_binary"]), row["label"])
                st.markdown(f"**Label Asli (Ground Truth):** `{true_label}`  |  **Skor Lexicon:** `{row['score']}`")

                with st.spinner("Menjalankan model..."):
                    result = run_prediction(str(row["hasil_preprocessing"]), vectorizer, nb_model, svm_model)
                show_prediction_result(result)

                c1, c2 = st.columns(2)
                with c1:
                    ok = result["nb_pred"] == true_label
                    st.markdown(f'<span class="{"badge-ok" if ok else "badge-bad"}">Naive Bayes: {"Benar" if ok else "Salah"}</span>', unsafe_allow_html=True)
                with c2:
                    ok = result["svm_pred"] == true_label
                    st.markdown(f'<span class="{"badge-ok" if ok else "badge-bad"}">SVM: {"Benar" if ok else "Salah"}</span>', unsafe_allow_html=True)
            else:
                st.info("Pilih salah satu baris di tabel, atau klik 'Pilih Acak' buat mulai.", icon=":material/touch_app:")

# =====================================================================================
# Aktifkan animasi scroll-reveal (harus dipanggil PALING TERAKHIR supaya semua
# elemen halaman sudah ada di DOM sebelum di-observe)
# =====================================================================================
inject_scroll_reveal()