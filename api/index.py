"""
Green Bond Classification API - Vercel Serverless Function
Bank Indonesia - DSta-DSMF
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import PyPDF2
import os
import io
import re
import hashlib
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

from classifier import (
    LABELS, LABEL_DISPLAY,
    GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS,
    calc_score, rule_based_classify,
    ModelManager,
    determine_final_classification, format_keyword_strings, get_confidence_level,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM
)

# Import R2 Storage
try:
    from .r2_storage import R2Storage
except ImportError:
    from r2_storage import R2Storage

# ============================================================================
# CONFIGURATION
# ============================================================================
CORS_ORIGINS = os.environ.get(
    'CORS_ORIGINS', '*'
).split(',')

API_KEY = os.environ.get('API_KEY')

app = FastAPI(
    title="Green Bond Classification API",
    description="API untuk mengklasifikasikan obligasi dari prospektus PDF (Serverless + R2)",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel puts files in /var/task
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ML_Models')

# Initialize Managers
model_manager = ModelManager()
model_manager.load(MODEL_DIR)
storage = R2Storage()

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
    text_search = text[:10000]

    patterns = [
        r'PROSPEKTUS[^\n]{0,100}?(?:PT\.?\s*)([A-Z][A-Za-z\s\.\,&]+?(?:\s*Tbk\.?)?)\s*(?:\n|$)',
        r'(?:Emiten|Penerbit)\s*[:\s]+(?:PT\.?\s*)?([A-Z][A-Za-z\s\.\,&]+?(?:\s*Tbk\.?)?)\s*(?:\n|$)',
        r'\b(PT\.?\s+[A-Z][A-Za-z\s\.\,&]+?\s+Tbk\.?)\b',
        r'(?:Nama\s+(?:Lengkap|Perusahaan|Emiten|Perseroan))\s*[:\s]+(?:PT\.?\s*)?([A-Z][A-Za-z\s\.\,&]+)',
        r'(?:Perseroan|Perusahaan)\s*[:\s]+(?:PT\.?\s*)?([A-Z][A-Za-z\s\.\,&]+)',
        r'\b(PT\.?\s+[A-Z][A-Za-z\s\.\,&]{5,50})\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_search, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip() if isinstance(match, str) else match[0].strip()
            name = re.sub(r'\s+', ' ', name)
            name = re.sub(r'[,\.\s]+$', '', name)
            name = name.strip()

            skip_words = ['prospektus', 'obligasi', 'sukuk', 'indonesia', 'tahun', 'seri', 'dengan']
            if len(name) < 5 or len(name) > 80:
                continue
            if any(sw in name.lower() for sw in skip_words):
                continue

            if not name.upper().startswith('PT'):
                name = 'PT ' + name

            return name.upper()

    fallback = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').strip()
    return fallback.upper()[:60]

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Dependency that checks API key for destructive endpoints."""
    if API_KEY is None:
        return  # auth disabled if env var not set
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/api")
async def api_root():
    """API root"""
    return {"message": "Green Bond Classification API", "version": "1.1.0", "status": "ready"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "storage_connected": storage.client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/classify")
async def classify_pdf(file: UploadFile = File(...), method: str = "hybrid"):
    """Classify a bond prospectus PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()

    # Save PDF to R2
    file_hash = hashlib.sha256(content).hexdigest()[:12]
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    file_id = f"{timestamp_str}_{file_hash}"
    r2_key = None

    if storage.client:
        r2_key = storage.save_pdf(content, f"{file_id}.pdf")

    text, pages = extract_text_from_pdf(content)
    word_count = len(text.split())
    company_name = extract_company_name(text, file.filename)

    rule_result = rule_based_classify(text)
    ml_result = model_manager.predict(text) if method in ['ml', 'hybrid'] else None

    final_label, final_confidence, final_method = determine_final_classification(rule_result, ml_result, method)

    label_info = LABEL_DISPLAY.get(final_label, LABEL_DISPLAY['obligasi_biasa'])

    green_unique = len(rule_result['keywords']['green'])
    sustain_unique = len(rule_result['keywords']['sustainability'])
    linked_unique = len(rule_result['keywords']['linked'])

    green_kw_str, sustain_kw_str, linked_kw_str = format_keyword_strings(rule_result)
    confidence_level = get_confidence_level(final_confidence)

    # Review logic
    needs_review = final_confidence < CONFIDENCE_THRESHOLD_HIGH
    review_reason = None
    if needs_review:
        if final_confidence < CONFIDENCE_THRESHOLD_MEDIUM:
            review_reason = "Low confidence - classification uncertain"
        else:
            review_reason = "Medium confidence - please verify"

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
        "review": {
            "needs_review": needs_review,
            "review_reason": review_reason
        },
        "document_info": {
            "pages": pages,
            "word_count": word_count,
            "char_count": len(text),
            "file_id": file_id,
            "pdf_url": f"/api/pdf/{file_id}" if r2_key else None
        },
        "scores": rule_result['scores'],
        "keyword_counts": {
            "green": sum([k['count'] for k in rule_result['keywords']['green']]),
            "sustainability": sum([k['count'] for k in rule_result['keywords']['sustainability']]),
            "linked": sum([k['count'] for k in rule_result['keywords']['linked']])
        },
        "unique_keywords": {"green": green_unique, "sustainability": sustain_unique, "linked": linked_unique},
        "keywords_found_str": {"green": green_kw_str, "sustainability": sustain_kw_str, "linked": linked_kw_str},
        "ml_probabilities": ml_result['probabilities'] if ml_result else None,
        "regulation": "POJK 18/2023",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/classify-bulk")
async def classify_bulk(files: List[UploadFile] = File(...), method: str = "hybrid"):
    """Classify multiple PDFs (Limited storage operations to prevent timeouts)"""
    results = []

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({"filename": file.filename, "error": "Not a PDF file"})
            continue

        try:
            content = await file.read()
            # Note: For bulk, we skip R2 storage to save time/bandwidth in serverless function
            text, pages = extract_text_from_pdf(content)
            word_count = len(text.split())
            company_name = extract_company_name(text, file.filename)

            rule_result = rule_based_classify(text)
            ml_result = model_manager.predict(text) if method in ['ml', 'hybrid'] else None

            final_label, final_confidence, final_method = determine_final_classification(rule_result, ml_result, method)
            label_info = LABEL_DISPLAY.get(final_label, LABEL_DISPLAY['obligasi_biasa'])
            confidence_level = get_confidence_level(final_confidence)

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
                "document_info": {"pages": pages, "word_count": word_count},
                "scores": rule_result['scores'],
                "status": "success",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e), "status": "failed"})

    return {"results": results, "total": len(results), "success": sum(1 for r in results if r.get('status') == 'success')}

@app.get("/api/labels")
async def get_labels():
    return {"labels": [{**info, 'key': key} for key, info in LABEL_DISPLAY.items()]}

# ============================================================================
# R2 STORAGE ENDPOINTS
# ============================================================================

class FeedbackRequest(BaseModel):
    filename: str
    original_label: str
    corrected_label: str
    confidence: float
    reason: Optional[str] = None
    company_name: Optional[str] = None
    scores: Optional[dict] = None
    confirmed: Optional[bool] = None

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest, _=Depends(verify_api_key)):
    """Submit user correction/feedback to R2"""
    if not storage.client:
        return {"status": "warning", "message": "Storage not configured, feedback not saved"}

    feedback_data = feedback.dict()
    feedback_data['timestamp'] = datetime.now().isoformat()

    success = storage.save_feedback(feedback_data)

    if success:
        return {"status": "success", "message": "Feedback saved to R2"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.get("/api/pdf/{file_id}")
async def get_pdf(file_id: str):
    """Retrieve PDF from R2"""
    if not storage.client:
        raise HTTPException(status_code=503, detail="Storage not configured")

    # Security check for file_id format
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    pdf_bytes = storage.get_pdf(f"{file_id}.pdf")

    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="PDF not found")

    from fastapi.responses import Response
    return Response(content=pdf_bytes, media_type="application/pdf")

# Handler for Vercel
handler = Mangum(app)
