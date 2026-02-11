"""
Green Bond Classification API
Bank Indonesia - DSta-DSMF
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import PyPDF2
import os
import io
import re
import hashlib
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
from contextlib import asynccontextmanager

from classifier import (
    LABELS, LABEL_DISPLAY,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM,
    GREEN_KEYWORDS, SUSTAINABILITY_KEYWORDS, SUSTAINABILITY_LINKED_KEYWORDS,
    calc_score, rule_based_classify,
    ModelManager,
    determine_final_classification, format_keyword_strings, get_confidence_level,
)

# ============================================================================
# CONFIGURATION (from env vars with safe defaults)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ML_Models')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
FEEDBACK_DB = os.path.join(BASE_DIR, 'ML_Dataset', 'feedback.db')
LEGACY_FEEDBACK_FILE = os.path.join(BASE_DIR, 'ML_Dataset', 'user_feedback.json')

# Security config
CORS_ORIGINS = os.environ.get(
    'CORS_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000'
).split(',')
MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 50 * 1024 * 1024))  # 50MB
API_KEY = os.environ.get('API_KEY')  # None = auth disabled (dev mode)

# Upload cleanup config
UPLOAD_MAX_AGE_DAYS = int(os.environ.get('UPLOAD_MAX_AGE_DAYS', 7))
UPLOAD_MAX_FILES = int(os.environ.get('UPLOAD_MAX_FILES', 200))

# ML model manager (replaces global model/vectorizer)
model_manager = ModelManager()

# ============================================================================
# UPLOAD CLEANUP
# ============================================================================
def cleanup_old_uploads():
    """Remove PDFs older than UPLOAD_MAX_AGE_DAYS and enforce max file count."""
    if not os.path.isdir(UPLOADS_DIR):
        return
    cutoff = time.time() - (UPLOAD_MAX_AGE_DAYS * 86400)
    files = []
    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        if os.path.isfile(fpath) and fname.lower().endswith('.pdf'):
            mtime = os.path.getmtime(fpath)
            if mtime < cutoff:
                os.remove(fpath)
            else:
                files.append((mtime, fpath))

    # Enforce max files (remove oldest first)
    files.sort(reverse=True)
    for _, fpath in files[UPLOAD_MAX_FILES:]:
        os.remove(fpath)


# ============================================================================
# SQLITE FEEDBACK
# ============================================================================
def _get_db_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(FEEDBACK_DB), exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            company_name TEXT,
            original_label TEXT NOT NULL,
            corrected_label TEXT NOT NULL,
            was_correct INTEGER NOT NULL,
            confidence REAL,
            reason TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _migrate_json_feedback():
    """One-time migration of legacy JSON feedback into SQLite."""
    import json
    if not os.path.exists(LEGACY_FEEDBACK_FILE):
        return
    try:
        with open(LEGACY_FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except (json.JSONDecodeError, IOError):
        return
    if not entries:
        return

    with _get_db_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if existing > 0:
            return  # already migrated
        for fb in entries:
            conn.execute(
                "INSERT INTO feedback (filename, company_name, original_label, corrected_label, was_correct, confidence, reason, timestamp) VALUES (?,?,?,?,?,?,?,?)",
                (
                    fb.get('filename', ''),
                    fb.get('company_name'),
                    fb.get('original_label', ''),
                    fb.get('corrected_label', ''),
                    1 if fb.get('was_correct') else 0,
                    fb.get('confidence'),
                    fb.get('reason'),
                    fb.get('timestamp', datetime.now().isoformat()),
                )
            )
        conn.commit()
    # Rename old file so it's not re-migrated
    try:
        os.rename(LEGACY_FEEDBACK_FILE, LEGACY_FEEDBACK_FILE + '.migrated')
    except OSError:
        pass


# ============================================================================
# SECURITY HELPERS
# ============================================================================
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Dependency that checks API key for destructive endpoints."""
    if API_KEY is None:
        return  # auth disabled in dev mode
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_safe_pdf_path(file_id: str) -> str:
    """Validate file_id and return safe absolute path, or raise 400."""
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")
    pdf_path = os.path.realpath(os.path.join(UPLOADS_DIR, f"{file_id}.pdf"))
    uploads_real = os.path.realpath(UPLOADS_DIR)
    if not pdf_path.startswith(uploads_real):
        raise HTTPException(status_code=400, detail="Invalid file ID")
    return pdf_path


# ============================================================================
# APP SETUP
# ============================================================================
@asynccontextmanager
async def lifespan(app):
    # Startup
    model_manager.load(MODEL_DIR)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    cleanup_old_uploads()
    _migrate_json_feedback()
    yield
    # Shutdown

app = FastAPI(
    title="Green Bond Classification API",
    description="API untuk mengklasifikasikan obligasi dari prospektus PDF",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - safe defaults (no wildcard + credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    fallback = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').strip()
    return fallback.upper()[:60]

# ============================================================================
# API ENDPOINTS
# ============================================================================

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
        "model_loaded": model_manager.is_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/classify")
async def classify_pdf(file: UploadFile = File(...), method: str = "hybrid"):
    """
    Classify a bond prospectus PDF

    - **file**: PDF file to classify
    - **method**: Classification method - 'rule', 'ml', or 'hybrid' (default)
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()

    # File size check
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )

    text, pages = extract_text_from_pdf(content)
    word_count = len(text.split())

    # Generate file ID with SHA-256 and save PDF
    file_hash = hashlib.sha256(content).hexdigest()[:12]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_id = f"{timestamp}_{file_hash}"
    pdf_path = os.path.join(UPLOADS_DIR, f"{file_id}.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(content)

    company_name = extract_company_name(text, file.filename)

    # Classify
    rule_result = rule_based_classify(text)
    ml_result = model_manager.predict(text) if method in ['ml', 'hybrid'] else None

    final_label, final_confidence, final_method = determine_final_classification(rule_result, ml_result, method)

    label_info = LABEL_DISPLAY.get(final_label, LABEL_DISPLAY['obligasi_biasa'])

    green_unique = len(rule_result['keywords']['green'])
    sustain_unique = len(rule_result['keywords']['sustainability'])
    linked_unique = len(rule_result['keywords']['linked'])

    green_kw_str, sustain_kw_str, linked_kw_str = format_keyword_strings(rule_result)

    confidence_level = get_confidence_level(final_confidence)

    # Human-in-the-Loop: Determine if needs review
    needs_review = final_confidence < CONFIDENCE_THRESHOLD_HIGH
    review_reason = None
    if needs_review:
        if final_confidence < CONFIDENCE_THRESHOLD_MEDIUM:
            review_reason = "Low confidence - classification uncertain"
        else:
            review_reason = "Medium confidence - please verify"

        max_score = max(rule_result['scores']['green'], rule_result['scores']['sustainability'], rule_result['scores']['linked'])
        if max_score < 10 and final_label != 'obligasi_biasa':
            review_reason = "No strong keyword signals but classified as non-regular bond"
            needs_review = True
        elif max_score >= 10 and final_label == 'obligasi_biasa':
            review_reason = "Has sustainability keywords but classified as regular bond"
            needs_review = True

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
            "review_reason": review_reason,
            "user_corrected": False,
            "corrected_label": None
        },
        "document_info": {
            "pages": pages,
            "word_count": word_count,
            "char_count": len(text),
            "file_id": file_id,
            "pdf_url": f"/api/pdf/{file_id}"
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

@app.post("/api/classify-bulk")
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

            if len(content) > MAX_UPLOAD_SIZE:
                results.append({
                    "filename": file.filename,
                    "error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB",
                    "status": "failed"
                })
                continue

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

            needs_review = final_confidence < CONFIDENCE_THRESHOLD_HIGH
            review_reason = None
            if needs_review:
                if final_confidence < CONFIDENCE_THRESHOLD_MEDIUM:
                    review_reason = "Low confidence - classification uncertain"
                else:
                    review_reason = "Medium confidence - please verify"

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
                "review": {
                    "needs_review": needs_review,
                    "review_reason": review_reason,
                    "user_corrected": False,
                    "corrected_label": None
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

# ============================================================================
# HUMAN-IN-THE-LOOP: Feedback System (SQLite)
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
    """Submit user correction/feedback for a classification."""
    with _get_db_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO feedback (filename, company_name, original_label, corrected_label, was_correct, confidence, reason, timestamp) VALUES (?,?,?,?,?,?,?,?)",
            (
                feedback.filename,
                feedback.company_name,
                feedback.original_label,
                feedback.corrected_label,
                1 if feedback.original_label == feedback.corrected_label else 0,
                feedback.confidence,
                feedback.reason,
                datetime.now().isoformat(),
            )
        )
        feedback_id = cursor.lastrowid
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    return {
        "status": "success",
        "message": "Feedback saved successfully",
        "feedback_id": feedback_id,
        "total_feedback": total
    }

# ============================================================================
# PDF VIEWER ENDPOINTS
# ============================================================================
@app.get("/api/pdf/{file_id}")
async def get_pdf(file_id: str):
    """Serve a previously uploaded PDF file"""
    pdf_path = get_safe_pdf_path(file_id)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={file_id}.pdf"}
    )

@app.delete("/api/pdf/{file_id}")
async def delete_pdf(file_id: str, _=Depends(verify_api_key)):
    """Delete a previously uploaded PDF file"""
    pdf_path = get_safe_pdf_path(file_id)

    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        return {"message": "PDF deleted successfully"}

    return {"message": "PDF not found or already deleted"}

@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Get statistics about user feedback"""
    with _get_db_conn() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()

    if not rows:
        return {
            "total_feedback": 0,
            "corrections_by_label": {},
            "accuracy_estimate": None,
            "needs_review_rate": None
        }

    corrections = {}
    correct_count = 0
    for row in rows:
        orig = row['original_label']
        corr = row['corrected_label']

        if orig not in corrections:
            corrections[orig] = {"total": 0, "corrected_to": {}}
        corrections[orig]["total"] += 1

        if orig != corr:
            if corr not in corrections[orig]["corrected_to"]:
                corrections[orig]["corrected_to"][corr] = 0
            corrections[orig]["corrected_to"][corr] += 1
        else:
            correct_count += 1

    total = len(rows)
    accuracy = correct_count / total if total else 0

    recent = [dict(row) for row in rows[:10]]

    return {
        "total_feedback": total,
        "correct_predictions": correct_count,
        "incorrect_predictions": total - correct_count,
        "accuracy_estimate": round(accuracy, 4),
        "corrections_by_label": corrections,
        "recent_feedback": recent
    }

@app.get("/api/feedback/export")
async def export_feedback():
    """Export all feedback as training data"""
    with _get_db_conn() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()

    training_data = []
    for row in rows:
        if row['corrected_label']:
            training_data.append({
                "filename": row['filename'],
                "label": row['corrected_label'],
                "source": "user_feedback",
                "original_prediction": row['original_label'],
                "confidence": row['confidence'],
                "timestamp": row['timestamp']
            })

    return {
        "total": len(training_data),
        "data": training_data
    }

@app.delete("/api/feedback/{feedback_id}")
async def delete_feedback(feedback_id: int, _=Depends(verify_api_key)):
    """Delete a specific feedback entry"""
    with _get_db_conn() as conn:
        conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))

    return {"status": "success", "message": f"Feedback {feedback_id} deleted"}

@app.post("/api/cleanup")
async def trigger_cleanup(_=Depends(verify_api_key)):
    """Manually trigger upload cleanup (auth-protected)"""
    cleanup_old_uploads()
    return {"status": "success", "message": "Cleanup completed"}

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
