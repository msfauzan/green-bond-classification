# Green Bond Classification

Aplikasi web untuk mengklasifikasikan obligasi berdasarkan **POJK 18/2023** (Penerbitan dan Persyaratan Efek Bersifat Utang dan/atau Sukuk Berwawasan Lingkungan).

## 🌿 Fitur

- **Upload PDF** - Single atau bulk upload prospektus obligasi
- **Auto Classification** - Klasifikasi otomatis menggunakan ML + Rule-based
- **4 Kategori**:
  - 🌿 Green Bond
  - ♻️ Sustainability Bond  
  - 🔗 Sustainability-Linked Bond
  - 📄 Obligasi Biasa
- **Detail Scoring** - Breakdown keyword scoring per kategori
- **Export XLSX** - Download hasil klasifikasi

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone repo
git clone https://github.com/msfauzan/green-bond-classification.git
cd green-bond-classification

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn webapp.api:app --host 0.0.0.0 --port 8000
```

Buka http://localhost:8000

## 📁 Structure

```
├── webapp/
│   ├── api.py              # FastAPI backend
│   └── static/
│       └── index.html      # Frontend UI
├── ML_Models/              # Trained ML models
├── ML_Dataset/             # Labeled dataset
├── generate_labels.py      # Pseudo-label generator
├── ml_trainer.py           # ML training script
└── requirements.txt
```

## 🔧 Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: HTML, TailwindCSS, Vanilla JS
- **ML**: scikit-learn (Logistic Regression, TF-IDF)
- **PDF**: PyPDF2

## 📊 Classification Logic

Menggunakan hybrid approach:
1. **Rule-based** - Keyword scoring dengan bobot berbeda
2. **ML Model** - Logistic Regression dengan TF-IDF features

Keywords berdasarkan POJK 18/2023 termasuk:
- `green bond`, `obligasi hijau`, `efek berwawasan lingkungan`
- `sustainability bond`, `proyek sosial`, `sdgs`
- `sustainability-linked`, `kpi keberlanjutan`, `step-up/down`

## 📄 License

MIT License
