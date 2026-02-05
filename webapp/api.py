"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GREEN BOND CLASSIFICATION API                              ║
║                Bank Indonesia - DSta-DSMF                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import joblib
import PyPDF2
import numpy as np
import os
import io
from datetime import datetime
from typing import Optional, List
import json
import re

app = FastAPI(
    title="Green Bond Classification API",
    description="API untuk mengklasifikasikan obligasi dari prospektus PDF",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ML_Models')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

# Labels
LABELS = ['green_bond', 'obligasi_biasa', 'sustainability_bond', 'sustainability_linked_bond']
LABEL_DISPLAY = {
    'green_bond': {'name': 'Green Bond', 'emoji': '🌿', 'color': '#22c55e'},
    'sustainability_bond': {'name': 'Sustainability Bond', 'emoji': '♻️', 'color': '#3b82f6'},
    'sustainability_linked_bond': {'name': 'Sustainability-Linked Bond', 'emoji': '🔗', 'color': '#8b5cf6'},
    'obligasi_biasa': {'name': 'Obligasi Biasa', 'emoji': '📄', 'color': '#6b7280'},
}

# Keywords for rule-based scoring (Updated with POJK 18/2023)
# POJK 18/2023: Penerbitan dan Persyaratan Efek Bersifat Utang dan/atau Sukuk Berwawasan Lingkungan
GREEN_KEYWORDS = [
    ('green bond', 10), ('green sukuk', 10), ('obligasi hijau', 10), ('sukuk hijau', 10),
    ('pojk 18/2023', 15), ('pojk 18 tahun 2023', 15), ('pojk18/2023', 15),
    ('pojk 60', 8), ('pojk60', 8), ('kubl', 8),
    ('efek berwawasan lingkungan', 12), ('ebl', 8),
    ('proyek hijau', 5), ('green project', 5),
    ('energi terbarukan', 3), ('renewable energy', 3),
    ('efisiensi energi', 3), ('energy efficiency', 3),
    ('pengelolaan air', 2), ('water management', 2),
    ('pencegahan polusi', 2), ('pollution prevention', 2),
    ('green building', 2), ('bangunan hijau', 2),
    ('transportasi ramah lingkungan', 2), ('clean transportation', 2),
    ('pengelolaan limbah', 2), ('waste management', 2),
    ('adaptasi perubahan iklim', 2), ('climate adaptation', 2),
]
SUSTAINABILITY_KEYWORDS = [
    ('sustainability bond', 10), ('sustainability sukuk', 10),
    ('obligasi keberlanjutan', 10), ('sukuk keberlanjutan', 10),
    ('ebus keberlanjutan', 10),
    ('proyek sosial', 5), ('social project', 5),
    ('social bond', 5), ('obligasi sosial', 5),
    ('sdgs', 3), ('sustainable development goals', 3),
    ('pembangunan berkelanjutan', 2), ('dampak sosial', 2),
    ('affordable housing', 2), ('perumahan terjangkau', 2),
    ('akses layanan kesehatan', 2), ('healthcare access', 2),
    ('pendidikan', 2), ('education', 2),
    ('ketahanan pangan', 2), ('food security', 2),
]
SUSTAINABILITY_LINKED_KEYWORDS = [
    ('sustainability linked bond', 10), ('sustainability-linked bond', 10),
    ('sustainability linked sukuk', 10), ('sustainability-linked sukuk', 10),
    ('sustainability performance target', 8), ('spt', 5),
    ('kpi keberlanjutan', 8), ('sustainability kpi', 8),
    ('step-up', 3), ('step-down', 3),
    ('penalty', 2), ('premium', 2),
    ('target kinerja', 2), ('performance target', 2),
]

# ============================================================================
# LOAD ML MODEL
# ============================================================================
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    try:
        # Find latest model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('logistic_regression') and f.endswith('.joblib')]
        vec_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('vectorizer') and f.endswith('.joblib')]
        
        if model_files and vec_files:
            model_files.sort(reverse=True)
            vec_files.sort(reverse=True)
            model = joblib.load(os.path.join(MODEL_DIR, model_files[0]))
            vectorizer = joblib.load(os.path.join(MODEL_DIR, vec_files[0]))
            print(f"✅ Loaded model: {model_files[0]}")
            return True
    except Exception as e:
        print(f"⚠️ Could not load ML model: {e}")
    return False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> tuple:
    """Extract text from PDF bytes"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ' '.join([page.extract_text() or '' for page in reader.pages])
        return text, len(reader.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_company_name(text: str, filename: str) -> str:
    """Extract company name from PDF text using patterns - returns UPPERCASE"""
    
    # Search in first 10000 chars for better coverage
    text_search = text[:10000]
    
    # Patterns ordered by reliability (most specific first)
    patterns = [
        # Pattern 1: "PROSPEKTUS ... PT XYZ Tbk" on title
        r'PROSPEKTUS[^\n]{0,100}?(?:PT\.?\s*)([A-Z][A-Za-z\s\.\,&]+?(?:\s*Tbk\.?)?)\s*(?:\n|$)',
        # Pattern 2: After "Emiten" or "Penerbit"
        r'(?:Emiten|Penerbit)\s*[:\s]+(?:PT\.?\s*)?([A-Z][A-Za-z\s\.\,&]+?(?:\s*Tbk\.?)?)\s*(?:\n|$)',
        # Pattern 3: "PT ... Tbk" standalone with Tbk
        r'\b(PT\.?\s+[A-Z][A-Za-z\s\.\,&]+?\s+Tbk\.?)\b',
        # Pattern 4: "Nama Lengkap" or "Nama Emiten"
        r'(?:Nama\s+(?:Lengkap|Perusahaan|Emiten|Perseroan))\s*[:\s]+(?:PT\.?\s*)?([A-Z][A-Za-z\s\.\,&]+)',
        # Pattern 5: Perseroan followed by name
        r'(?:Perseroan|Perusahaan)\s*[:\s]+(?:PT\.?\s*)?([A-Z][A-Za-z\s\.\,&]+)',
        # Pattern 6: Generic PT pattern
        r'\b(PT\.?\s+[A-Z][A-Za-z\s\.\,&]{5,50})\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_search, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip() if isinstance(match, str) else match[0].strip()
            # Clean up
            name = re.sub(r'\s+', ' ', name)
            name = re.sub(r'[,\.\s]+$', '', name)
            name = name.strip()
            
            # Skip if too short or contains bad words
            skip_words = ['prospektus', 'obligasi', 'sukuk', 'indonesia', 'tahun', 'seri', 'dengan']
            if len(name) < 5 or len(name) > 80:
                continue
            if any(sw in name.lower() for sw in skip_words):
                continue
            
            # Normalize PT prefix
            if not name.upper().startswith('PT'):
                name = 'PT ' + name
            
            return name.upper()  # Return UPPERCASE
    
    # Fallback: try to extract from filename
    fn_patterns = [
        r'(?:prospektus|obligasi|sukuk)[_\-\s]+(.+?)(?:[_\-\s]+(?:\d{4}|tahun|seri)|$)',
        r'^(.+?)(?:[_\-\s]+(?:\d{4}|tahun|seri)|\.pdf$)',
    ]
    
    clean_fn = filename.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ').strip()
    for pattern in fn_patterns:
        match = re.search(pattern, clean_fn, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if len(name) > 2 and name.lower() not in ['prospektus', 'obligasi', 'sukuk']:
                name = re.sub(r'\s+', ' ', name).strip()
                if not name.upper().startswith('PT'):
                    name = 'PT ' + name
                return name.upper()
    
    # Last resort: use filename
    fallback = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').strip()
    return fallback.upper()[:60]

def calc_score(text: str, keywords: list) -> tuple:
    """Calculate keyword score"""
    text_lower = text.lower()
    total = 0
    found = []
    for kw, weight in keywords:
        count = text_lower.count(kw.lower())
        if count > 0:
            total += weight * min(count, 5)
            found.append({'keyword': kw, 'count': count, 'weight': weight})
    return total, found

def rule_based_classify(text: str) -> dict:
    """Rule-based classification with keyword scoring"""
    green_score, green_kw = calc_score(text, GREEN_KEYWORDS)
    sustain_score, sustain_kw = calc_score(text, SUSTAINABILITY_KEYWORDS)
    linked_score, linked_kw = calc_score(text, SUSTAINABILITY_LINKED_KEYWORDS)
    
    # Determine label
    if linked_score >= 10 and linked_score >= sustain_score and linked_score >= green_score:
        label = 'sustainability_linked_bond'
    elif sustain_score >= 10 and sustain_score >= green_score:
        label = 'sustainability_bond'
    elif green_score >= 10:
        label = 'green_bond'
    else:
        label = 'obligasi_biasa'
    
    # Confidence based on score magnitude
    max_score = max(green_score, sustain_score, linked_score)
    if max_score >= 50:
        confidence = 0.95
    elif max_score >= 30:
        confidence = 0.85
    elif max_score >= 10:
        confidence = 0.70
    else:
        confidence = 0.50
    
    return {
        'label': label,
        'confidence': confidence,
        'scores': {
            'green': green_score,
            'sustainability': sustain_score,
            'linked': linked_score
        },
        'keywords': {
            'green': green_kw,
            'sustainability': sustain_kw,
            'linked': linked_kw
        }
    }

def ml_classify(text: str) -> dict:
    """ML-based classification"""
    if model is None or vectorizer is None:
        return None
    
    try:
        # TF-IDF features
        tfidf = vectorizer.transform([text])
        
        # Get keyword scores as additional features
        green_score, _ = calc_score(text, GREEN_KEYWORDS)
        sustain_score, _ = calc_score(text, SUSTAINABILITY_KEYWORDS)
        linked_score, _ = calc_score(text, SUSTAINABILITY_LINKED_KEYWORDS)
        
        # Word count
        word_count = len(text.split())
        
        # Combine features
        extra_features = np.array([[green_score, sustain_score, linked_score, word_count, 0]])
        from scipy.sparse import hstack
        features = hstack([tfidf, extra_features])
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return {
            'label': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(model.classes_, probabilities)
            }
        }
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    load_model()
    os.makedirs(STATIC_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    html_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Green Bond Classification API</h1><p>Upload endpoint: POST /classify</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/classify")
async def classify_pdf(file: UploadFile = File(...), method: str = "hybrid"):
    """
    Classify a bond prospectus PDF
    
    - **file**: PDF file to classify
    - **method**: Classification method - 'rule', 'ml', or 'hybrid' (default)
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read file
    content = await file.read()
    text, pages = extract_text_from_pdf(content)
    word_count = len(text.split())
    
    # Extract company name
    company_name = extract_company_name(text, file.filename)
    
    # Classify
    rule_result = rule_based_classify(text)
    ml_result = ml_classify(text) if method in ['ml', 'hybrid'] else None
    
    # Determine final result
    if method == 'rule' or ml_result is None:
        final_label = rule_result['label']
        final_confidence = rule_result['confidence']
        final_method = 'rule-based'
    elif method == 'ml':
        final_label = ml_result['label']
        final_confidence = ml_result['confidence']
        final_method = 'machine-learning'
    else:  # hybrid
        # Use ML if confident, otherwise use rule-based
        if ml_result['confidence'] >= 0.8:
            final_label = ml_result['label']
            final_confidence = ml_result['confidence']
            final_method = 'machine-learning'
        else:
            final_label = rule_result['label']
            final_confidence = rule_result['confidence']
            final_method = 'rule-based'
    
    label_info = LABEL_DISPLAY.get(final_label, LABEL_DISPLAY['obligasi_biasa'])
    
    # Count unique keywords
    green_unique = len(rule_result['keywords']['green'])
    sustain_unique = len(rule_result['keywords']['sustainability'])
    linked_unique = len(rule_result['keywords']['linked'])
    
    # Format keywords found as strings
    green_kw_str = '; '.join([f"{k['keyword']} ({k['count']}x{k['weight']}={k['count']*k['weight']})" for k in rule_result['keywords']['green']])
    sustain_kw_str = '; '.join([f"{k['keyword']} ({k['count']}x{k['weight']}={k['count']*k['weight']})" for k in rule_result['keywords']['sustainability']])
    linked_kw_str = '; '.join([f"{k['keyword']} ({k['count']}x{k['weight']}={k['count']*k['weight']})" for k in rule_result['keywords']['linked']])
    
    # Determine confidence level
    if final_confidence >= 0.8:
        confidence_level = 'high'
    elif final_confidence >= 0.5:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'
    
    return {
        "filename": file.filename,
        "company_name": company_name,
        "classification": {
            "label": final_label,
            "display_name": label_info['name'],
            "emoji": label_info['emoji'],
            "color": label_info['color'],
            "confidence": round(final_confidence, 3),
            "confidence_level": confidence_level,
            "method": final_method
        },
        "document_info": {
            "pages": pages,
            "word_count": word_count,
            "char_count": len(text)
        },
        "scores": rule_result['scores'],
        "keyword_counts": {
            "green": sum([k['count'] for k in rule_result['keywords']['green']]),
            "sustainability": sum([k['count'] for k in rule_result['keywords']['sustainability']]),
            "linked": sum([k['count'] for k in rule_result['keywords']['linked']])
        },
        "unique_keywords": {
            "green": green_unique,
            "sustainability": sustain_unique,
            "linked": linked_unique
        },
        "keywords_found_str": {
            "green": green_kw_str,
            "sustainability": sustain_kw_str,
            "linked": linked_kw_str
        },
        "keywords_found": rule_result['keywords'],
        "ml_probabilities": ml_result['probabilities'] if ml_result else None,
        "regulation": "POJK 18/2023",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/classify-bulk")
async def classify_bulk(files: List[UploadFile] = File(...), method: str = "hybrid"):
    """
    Classify multiple bond prospectus PDFs
    
    - **files**: Multiple PDF files to classify
    - **method**: Classification method - 'rule', 'ml', or 'hybrid' (default)
    """
    results = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({"filename": file.filename, "error": "Not a PDF file"})
            continue
        
        try:
            content = await file.read()
            text, pages = extract_text_from_pdf(content)
            word_count = len(text.split())
            company_name = extract_company_name(text, file.filename)
            
            rule_result = rule_based_classify(text)
            ml_result = ml_classify(text) if method in ['ml', 'hybrid'] else None
            
            if method == 'rule' or ml_result is None:
                final_label = rule_result['label']
                final_confidence = rule_result['confidence']
                final_method = 'rule-based'
            elif method == 'ml':
                final_label = ml_result['label']
                final_confidence = ml_result['confidence']
                final_method = 'machine-learning'
            else:
                if ml_result['confidence'] >= 0.8:
                    final_label = ml_result['label']
                    final_confidence = ml_result['confidence']
                    final_method = 'machine-learning'
                else:
                    final_label = rule_result['label']
                    final_confidence = rule_result['confidence']
                    final_method = 'rule-based'
            
            label_info = LABEL_DISPLAY.get(final_label, LABEL_DISPLAY['obligasi_biasa'])
            
            # Count unique keywords
            green_unique = len(rule_result['keywords']['green'])
            sustain_unique = len(rule_result['keywords']['sustainability'])
            linked_unique = len(rule_result['keywords']['linked'])
            
            # Format keywords found as strings
            green_kw_str = '; '.join([f"{k['keyword']} ({k['count']}x{k['weight']}={k['count']*k['weight']})" for k in rule_result['keywords']['green']])
            sustain_kw_str = '; '.join([f"{k['keyword']} ({k['count']}x{k['weight']}={k['count']*k['weight']})" for k in rule_result['keywords']['sustainability']])
            linked_kw_str = '; '.join([f"{k['keyword']} ({k['count']}x{k['weight']}={k['count']*k['weight']})" for k in rule_result['keywords']['linked']])
            
            if final_confidence >= 0.8:
                confidence_level = 'high'
            elif final_confidence >= 0.5:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            results.append({
                "filename": file.filename,
                "company_name": company_name,
                "classification": {
                    "label": final_label,
                    "display_name": label_info['name'],
                    "emoji": label_info['emoji'],
                    "color": label_info['color'],
                    "confidence": round(final_confidence, 3),
                    "confidence_level": confidence_level,
                    "method": final_method
                },
                "document_info": {
                    "pages": pages,
                    "word_count": word_count,
                    "char_count": len(text)
                },
                "scores": rule_result['scores'],
                "keyword_counts": {
                    "green": sum([k['count'] for k in rule_result['keywords']['green']]),
                    "sustainability": sum([k['count'] for k in rule_result['keywords']['sustainability']]),
                    "linked": sum([k['count'] for k in rule_result['keywords']['linked']])
                },
                "unique_keywords": {
                    "green": green_unique,
                    "sustainability": sustain_unique,
                    "linked": linked_unique
                },
                "keywords_found_str": {
                    "green": green_kw_str,
                    "sustainability": sustain_kw_str,
                    "linked": linked_kw_str
                },
                "regulation": "POJK 18/2023",
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e), "status": "failed"})
    
    return {"results": results, "total": len(results), "success": sum(1 for r in results if r.get('status') == 'success')}

@app.get("/labels")
async def get_labels():
    """Get all classification labels"""
    return {
        "labels": [
            {**info, 'key': key}
            for key, info in LABEL_DISPLAY.items()
        ]
    }

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
