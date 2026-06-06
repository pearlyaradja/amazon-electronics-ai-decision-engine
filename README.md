# Amazon Electronics — AI Decision Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Apache_Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

**Final Project Mata Kuliah Big Data & AI**  
Sistem cerdas untuk menganalisis jutaan ulasan pelanggan Amazon Electronics dan menghasilkan keputusan otomatis (*Pertahankan / Evaluasi / Tarik*) untuk setiap produk.

Proyek ini menggabungkan dua pendekatan:
1. **Review Quality Filter** — Menyaring ulasan spam/tidak reliabel menggunakan XGBoost & Random Forest Classifier.
2. **Aspect-Based Sentiment Analysis (ABSA)** — Analisis sentimen menggunakan DistilBERT per aspek spesifik (Harga, Kualitas, Pengiriman, Layanan).

---

## 👥 Tim & Pembagian Tugas (2 Orang)

| Role | Anggota | Tanggung Jawab Utama |
|---|---|---|
| **Data & ML Engineer** | **Person A** | 1. Setup PySpark & Preprocessing Data (`pyspark`).<br>2. Training & Evaluasi **Model 1: Review Quality Filter** (`xgboost`/`scikit-learn`).<br>3. Training **Model Baseline: ABSA** (`TF-IDF + SVM`) untuk pembanding. |
| **NLP & Analytics Engineer** | **Person B** | 1. Implementasi **Model 2: ABSA (DistilBERT)**.<br>2. Pembuatan **Scoring Engine** (Agregasi skor per produk).<br>3. Pembuatan **Decision Engine** (Rule-based decision).<br>4. Visualisasi & Evaluasi perbandingan model. |

---

## 📁 Struktur Repositori

```
amazon-electronics-ai-decision-engine/
│
├── README.md                          # Dokumentasi proyek
├── requirements.txt                   # Dependencies Python
├── .gitignore                         # Mengabaikan file besar/temporary
├── config.py                          # Parameter konfigurasi global & path
│
├── data/
│   ├── raw/                           # Dataset mentah (.zip/.tsv) - Excluded
│   └── processed/                     # Output parquet dari Step 1 - Excluded
│
├── notebooks/                         # Eksplorasi interaktif & model development
│   ├── 01_preprocessing_eda.ipynb     # Preprocessing Spark & EDA [Person A]
│   ├── 02_baseline_absa.ipynb         # ML Baseline untuk ABSA [Person A]
│   └── 06_visualization_report.ipynb  # Visualisasi report akhir [Person B]
│
├── src/                               # Source code utama pipeline
│   ├── __init__.py
│   ├── step1_preprocessing/           # Preprocessing script helper
│   ├── step2_quality_filter/          # Model Quality Filter (XGBoost/RF)
│   ├── step3_absa/                    # Model ABSA (DistilBERT)
│   ├── step4_scoring/                 # Aggregator skor produk
│   ├── step5_decision/                # Decision Engine
│   └── step6_visualization/           # Generator charts & confusion matrix
│
├── models/                            # Model biner yang disimpan (XGBoost, SVM, dll)
├── outputs/                           # Folder output figures dan keputusan CSV
└── tests/                             # File unit testing
```

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Prasyarat (Prerequisites)
- **Python 3.9+**
- **Java Development Kit (JDK) 8 atau 11** (diperlukan oleh Apache Spark)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Direktori
Jalankan file config sekali untuk menginisialisasi semua folder yang diperlukan:
```bash
python config.py
```
*(Catatan: Letakkan file `amazon_reviews_us_Electronics_v1_00.tsv` di folder `dataset/` sesuai dengan konfigurasi path)*
