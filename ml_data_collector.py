"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ML DATA COLLECTOR - Green Bond Classification                       ║
║           Bank Indonesia - DSta-DSMF                                         ║
║                                                                              ║
║  Features:                                                                   ║
║  - Extract raw text for ML training                                          ║
║  - Comprehensive feature extraction                                          ║
║  - Confidence scoring for manual review prioritization                       ║
║  - Export to ML-ready formats (CSV, JSON, JSONL for BERT)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import time
from datetime import datetime
from pathlib import Path

import PyPDF2
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = r"d:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
PDF_DIR = os.path.join(BASE_DIR, "Prospektus_Downloaded")
OUTPUT_DIR = os.path.join(BASE_DIR, "ML_Dataset")

# Output files
DATASET_CSV = os.path.join(OUTPUT_DIR, "prospektus_dataset.csv")
DATASET_JSON = os.path.join(OUTPUT_DIR, "prospektus_dataset.json")
DATASET_JSONL = os.path.join(OUTPUT_DIR, "prospektus_for_bert.jsonl")  # For transformers
RAW_TEXT_DIR = os.path.join(OUTPUT_DIR, "raw_texts")
REVIEW_PRIORITY = os.path.join(OUTPUT_DIR, "manual_review_priority.xlsx")

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED KEYWORDS dengan WEIGHTED SCORING
# ═══════════════════════════════════════════════════════════════════════════════

# Format: (keyword, weight)
# Weight: 10 = definitive, 5 = strong indicator, 2 = supporting, 1 = weak

GREEN_KEYWORDS = [
    # Definitive (10) - Jika ada ini, pasti Green Bond
    ("green bond", 10),
    ("green sukuk", 10),
    ("obligasi hijau", 10),
    ("sukuk hijau", 10),
    ("efek bersifat utang berwawasan lingkungan", 10),
    
    # Regulatory Terms (8)
    ("kubl", 8),  # Kegiatan Usaha Berwawasan Lingkungan
    ("pojk 60", 5),
    ("pojk no. 60", 5),
    
    # Strong Indicators (5)
    ("proyek hijau", 5),
    ("proyek ramah lingkungan", 5),
    ("green project", 5),
    ("eligible green project", 5),
    
    # Supporting (2) - Perlu kombinasi dengan yang lain
    ("energi terbarukan", 2),
    ("rendah karbon", 2),
    ("low carbon", 2),
    ("renewable energy", 2),
    ("efisiensi energi", 2),
    ("energy efficiency", 2),
    ("green building", 2),
    ("bangunan hijau", 2),
]

SUSTAINABILITY_KEYWORDS = [
    # Definitive (10)
    ("sustainability bond", 10),
    ("sustainability sukuk", 10),
    ("obligasi keberlanjutan", 10),
    ("sukuk keberlanjutan", 10),
    ("efek bersifat utang keberlanjutan", 10),
    ("ebus keberlanjutan", 10),
    
    # Regulatory Terms (8)
    ("pojk 18", 5),
    ("pojk no. 18", 5),
    
    # Strong Indicators (5)
    ("proyek sosial", 5),
    ("social project", 5),
    ("proyek lingkungan dan sosial", 5),
    ("environmental and social", 5),
    
    # Supporting (2)
    ("sdgs", 2),
    ("sustainable development goals", 2),
    ("pembangunan berkelanjutan", 2),
    ("dampak sosial", 2),
    ("social impact", 2),
]

SUSTAINABILITY_LINKED_KEYWORDS = [
    # Definitive (10)
    ("sustainability linked bond", 10),
    ("sustainability-linked bond", 10),
    ("sustainability linked sukuk", 10),
    ("sustainability-linked sukuk", 10),
    ("efek bersifat utang terkait keberlanjutan", 10),
    ("ebus terkait keberlanjutan", 10),
    
    # Strong Indicators (8) - KPI related
    ("sustainability performance target", 8),
    # ("spt", 5),  # Sustainability Performance Target
    ("key performance indicator", 5),
    ("kpi keberlanjutan", 8),
    ("indikator kinerja utama keberlanjutan", 8),
    
    # Supporting (3)
    ("step-up", 3),
    ("step-down", 3),
    ("coupon adjustment", 3),
    ("penalty", 2),
    ("target pencapaian", 3),
]

# Exclusion keywords - reduce score if these appear without main keywords
EXCLUSION_CONTEXT = [
    "tidak termasuk green bond",
    "bukan green bond",
    "bukan merupakan green",
    "tidak diklasifikasikan sebagai green",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path):
    """Extract full text from PDF."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"
            return full_text, len(pdf_reader.pages)
    except Exception as e:
        return "", 0


def calculate_weighted_score(text, keywords_with_weights):
    """Calculate weighted score based on keyword matches."""
    text_lower = text.lower()
    total_score = 0
    matches = []
    
    for keyword, weight in keywords_with_weights:
        count = text_lower.count(keyword.lower())
        if count > 0:
            # Diminishing returns for repeated keywords
            score = weight * min(count, 5)  # Cap at 5 occurrences
            total_score += score
            matches.append({
                'keyword': keyword,
                'count': count,
                'weight': weight,
                'score': score
            })
    
    return total_score, matches


def check_exclusions(text):
    """Check for exclusion patterns that indicate NOT a green/sustainability bond."""
    text_lower = text.lower()
    for pattern in EXCLUSION_CONTEXT:
        if pattern in text_lower:
            return True
    return False


def extract_metadata(text, filename):
    """Extract metadata from prospektus text."""
    metadata = {
        'stock_code': '',
        'company_name': '',
        'bond_value': '',
        'maturity': '',
        'coupon_rate': '',
    }
    
    # Extract stock code from filename
    code_match = re.search(r'\[([A-Z]{4})\s*\]', filename)
    if code_match:
        metadata['stock_code'] = code_match.group(1)
    
    # Try to extract bond value (nominal)
    value_patterns = [
        r'sebesar\s+Rp\s*([\d.,]+)\s*(miliar|triliun|juta)',
        r'nominal\s+Rp\s*([\d.,]+)',
        r'pokok\s+obligasi.*?Rp\s*([\d.,]+)',
    ]
    for pattern in value_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['bond_value'] = match.group(0)
            break
    
    # Try to extract maturity
    maturity_patterns = [
        r'jatuh tempo.*?(\d+)\s*(tahun|bulan)',
        r'tenor.*?(\d+)\s*(tahun|bulan)',
        r'jangka waktu.*?(\d+)\s*(tahun|bulan)',
    ]
    for pattern in maturity_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata['maturity'] = f"{match.group(1)} {match.group(2)}"
            break
    
    return metadata


def extract_comprehensive_features(pdf_path):
    """Extract all features from a PDF for ML training."""
    filename = os.path.basename(pdf_path)
    
    # Initialize features
    features = {
        'filename': filename,
        'filepath': pdf_path,
        'processed_at': datetime.now().isoformat(),
        
        # Basic stats
        'total_pages': 0,
        'total_words': 0,
        'total_chars': 0,
        
        # Weighted scores
        'green_score': 0,
        'sustainability_score': 0,
        'sustainability_linked_score': 0,
        
        # Raw keyword counts (total occurrences)
        'green_keyword_count': 0,
        'sustainability_keyword_count': 0,
        'sustainability_linked_keyword_count': 0,
        
        # Unique keyword types found
        'green_unique_keywords': 0,
        'sustainability_unique_keywords': 0,
        'sustainability_linked_unique_keywords': 0,
        
        # Classification
        'predicted_class': 'obligasi_biasa',
        'confidence': 0.0,
        'confidence_level': 'low',
        'needs_review': True,
        
        # Metadata
        'stock_code': '',
        'bond_value': '',
        'maturity': '',
        
        # Has exclusion context
        'has_exclusion': False,
        
        # Keyword matches (for debugging)
        'green_matches': [],
        'sustainability_matches': [],
        'sustainability_linked_matches': [],
        
        # Raw text (truncated for CSV, full in JSON)
        'text_preview': '',
        'raw_text_file': '',
        
        # Error handling
        'status': 'success',
        'error': '',
    }
    
    try:
        # Extract text
        full_text, num_pages = extract_text_from_pdf(pdf_path)
        
        if not full_text:
            features['status'] = 'error'
            features['error'] = 'Could not extract text'
            return features, ""
        
        features['total_pages'] = num_pages
        features['total_words'] = len(full_text.split())
        features['total_chars'] = len(full_text)
        features['text_preview'] = full_text[:500].replace('\n', ' ')
        
        # Calculate weighted scores
        green_score, green_matches = calculate_weighted_score(full_text, GREEN_KEYWORDS)
        sustain_score, sustain_matches = calculate_weighted_score(full_text, SUSTAINABILITY_KEYWORDS)
        linked_score, linked_matches = calculate_weighted_score(full_text, SUSTAINABILITY_LINKED_KEYWORDS)
        
        features['green_score'] = green_score
        features['sustainability_score'] = sustain_score
        features['sustainability_linked_score'] = linked_score
        
        features['green_keyword_count'] = sum(m['count'] for m in green_matches)
        features['sustainability_keyword_count'] = sum(m['count'] for m in sustain_matches)
        features['sustainability_linked_keyword_count'] = sum(m['count'] for m in linked_matches)
        
        # Unique keyword types found (how many different keywords matched)
        features['green_unique_keywords'] = len(green_matches)
        features['sustainability_unique_keywords'] = len(sustain_matches)
        features['sustainability_linked_unique_keywords'] = len(linked_matches)
        
        features['green_matches'] = json.dumps(green_matches)
        features['sustainability_matches'] = json.dumps(sustain_matches)
        features['sustainability_linked_matches'] = json.dumps(linked_matches)
        
        # Check exclusions
        features['has_exclusion'] = check_exclusions(full_text)
        
        # Extract metadata
        metadata = extract_metadata(full_text, filename)
        features.update(metadata)
        
        # Classification with confidence
        features = classify_with_confidence(features)
        
        return features, full_text
        
    except Exception as e:
        features['status'] = 'error'
        features['error'] = str(e)
        return features, ""


def classify_with_confidence(features):
    """Classify document and calculate confidence score."""
    green = features['green_score']
    sustain = features['sustainability_score']
    linked = features['sustainability_linked_score']
    
    total_score = green + sustain + linked
    max_score = max(green, sustain, linked)
    
    # Apply exclusion penalty
    if features['has_exclusion']:
        green *= 0.3
        sustain *= 0.3
        linked *= 0.3
    
    # Classification logic
    if max_score == 0:
        features['predicted_class'] = 'obligasi_biasa'
        features['confidence'] = 0.95  # Very confident it's regular bond
        features['confidence_level'] = 'high'
        features['needs_review'] = False
        
    elif linked >= 10 and linked >= sustain and linked >= green:
        features['predicted_class'] = 'sustainability_linked_bond'
        features['confidence'] = min(linked / 30, 1.0)  # Normalize
        
    elif sustain >= 10 and sustain >= green:
        features['predicted_class'] = 'sustainability_bond'
        features['confidence'] = min(sustain / 30, 1.0)
        
    elif green >= 10:
        features['predicted_class'] = 'green_bond'
        features['confidence'] = min(green / 30, 1.0)
        
    else:
        # Low scores - uncertain
        features['predicted_class'] = 'uncertain'
        features['confidence'] = max_score / 30
    
    # Determine confidence level and review need
    if features['confidence'] >= 0.7:
        features['confidence_level'] = 'high'
        features['needs_review'] = False
    elif features['confidence'] >= 0.4:
        features['confidence_level'] = 'medium'
        features['needs_review'] = True
    else:
        features['confidence_level'] = 'low'
        features['needs_review'] = True
    
    # Force review if it's a potential green/sustainability bond
    if features['predicted_class'] != 'obligasi_biasa' and features['confidence'] < 0.8:
        features['needs_review'] = True
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_raw_text(filename, text, output_dir):
    """Save raw text to file for ML training."""
    os.makedirs(output_dir, exist_ok=True)
    text_filename = os.path.splitext(filename)[0] + '.txt'
    text_path = os.path.join(output_dir, text_filename)
    
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text_filename


def export_for_bert(records, output_path):
    """Export dataset in JSONL format for BERT/transformers training."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            # Format for transformers
            bert_record = {
                'text': record.get('text_preview', '')[:512],  # BERT max length
                'label': record['predicted_class'],
                'label_id': {
                    'obligasi_biasa': 0,
                    'green_bond': 1,
                    'sustainability_bond': 2,
                    'sustainability_linked_bond': 3,
                    'uncertain': 4
                }.get(record['predicted_class'], 0),
                'filename': record['filename'],
                'confidence': record['confidence'],
                'needs_review': record['needs_review'],
            }
            f.write(json.dumps(bert_record, ensure_ascii=False) + '\n')


def create_review_priority_list(records, output_path):
    """Create prioritized list for manual review."""
    review_records = [r for r in records if r['needs_review']]
    
    # Sort by: 1) potential green/sustainability first, 2) then by confidence (lower = more uncertain)
    def priority_key(r):
        class_priority = {
            'green_bond': 0,
            'sustainability_bond': 1,
            'sustainability_linked_bond': 2,
            'uncertain': 3,
            'obligasi_biasa': 4
        }
        return (class_priority.get(r['predicted_class'], 5), r['confidence'])
    
    review_records.sort(key=priority_key)
    
    # Create DataFrame with relevant columns
    df = pd.DataFrame(review_records)
    columns = [
        'filename', 'predicted_class', 'confidence', 'confidence_level',
        'green_score', 'sustainability_score', 'sustainability_linked_score',
        'stock_code', 'total_pages', 'text_preview'
    ]
    existing_cols = [c for c in columns if c in df.columns]
    df = df[existing_cols]
    
    # Add manual review columns
    df['manual_label'] = ''
    df['reviewer_notes'] = ''
    df['reviewed'] = False
    
    df.to_excel(output_path, index=False)
    return len(review_records)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_all_pdfs(pdf_dir=PDF_DIR, output_dir=OUTPUT_DIR):
    """Process all PDFs and create ML-ready dataset."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           ML DATA COLLECTOR - Green Bond Classification                       ║
    ║           Bank Indonesia - DSta-DSMF                                         ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(RAW_TEXT_DIR, exist_ok=True)
    
    # Get all PDFs
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"📂 Found {len(pdf_files)} PDF files in {pdf_dir}")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 70)
    
    all_records = []
    class_counts = {
        'green_bond': 0,
        'sustainability_bond': 0,
        'sustainability_linked_bond': 0,
        'obligasi_biasa': 0,
        'uncertain': 0
    }
    
    for idx, filename in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_dir, filename)
        
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {filename[:50]}...")
        
        # Extract features
        features, full_text = extract_comprehensive_features(pdf_path)
        
        # Save raw text
        if full_text:
            text_file = save_raw_text(filename, full_text, RAW_TEXT_DIR)
            features['raw_text_file'] = text_file
        
        # Print result
        pred_class = features['predicted_class']
        confidence = features['confidence']
        conf_level = features['confidence_level']
        
        emoji = {
            'green_bond': '🌿',
            'sustainability_bond': '♻️',
            'sustainability_linked_bond': '🔗',
            'obligasi_biasa': '📄',
            'uncertain': '❓'
        }.get(pred_class, '❓')
        
        print(f"   {emoji} {pred_class} (confidence: {confidence:.2f} - {conf_level})")
        
        if features['green_score'] > 0 or features['sustainability_score'] > 0 or features['sustainability_linked_score'] > 0:
            print(f"      Scores: Green={features['green_score']}, Sustain={features['sustainability_score']}, Linked={features['sustainability_linked_score']}")
        
        if features['needs_review']:
            print(f"      ⚠️ Needs manual review")
        
        all_records.append(features)
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
    
    # ===== EXPORT DATASETS =====
    print("\n" + "=" * 70)
    print("💾 EXPORTING DATASETS...")
    print("=" * 70)
    
    # 1. CSV (main dataset)
    df = pd.DataFrame(all_records)
    # Remove complex columns for CSV
    csv_columns = [c for c in df.columns if c not in ['green_matches', 'sustainability_matches', 'sustainability_linked_matches']]
    df[csv_columns].to_csv(DATASET_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ CSV: {DATASET_CSV}")
    
    # 2. JSON (full dataset with all details)
    with open(DATASET_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON: {DATASET_JSON}")
    
    # 3. JSONL for BERT
    export_for_bert(all_records, DATASET_JSONL)
    print(f"✅ JSONL (BERT): {DATASET_JSONL}")
    
    # 4. Manual review priority list
    review_count = create_review_priority_list(all_records, REVIEW_PRIORITY)
    print(f"✅ Review List: {REVIEW_PRIORITY} ({review_count} items)")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"\n📁 Total documents: {len(all_records)}")
    print("\nBy predicted class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_records) * 100
        print(f"   {cls}: {count} ({pct:.1f}%)")
    
    needs_review = sum(1 for r in all_records if r['needs_review'])
    print(f"\n⚠️ Documents needing manual review: {needs_review}")
    
    high_conf = sum(1 for r in all_records if r['confidence_level'] == 'high')
    med_conf = sum(1 for r in all_records if r['confidence_level'] == 'medium')
    low_conf = sum(1 for r in all_records if r['confidence_level'] == 'low')
    print(f"\nConfidence distribution:")
    print(f"   High: {high_conf}")
    print(f"   Medium: {med_conf}")
    print(f"   Low: {low_conf}")
    
    print("\n" + "=" * 70)
    print("✅ ML DATASET COLLECTION COMPLETE!")
    print("=" * 70)
    
    print(f"""
📁 Output files:
   - {DATASET_CSV} (main dataset)
   - {DATASET_JSON} (full details)
   - {DATASET_JSONL} (for BERT training)
   - {REVIEW_PRIORITY} (manual review list)
   - {RAW_TEXT_DIR}/ (raw text files)

🎯 Next steps:
   1. Review documents in '{os.path.basename(REVIEW_PRIORITY)}'
   2. Fill in 'manual_label' column
   3. Use labeled data for ML training
    """)
    
    return all_records


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Data Collector for Green Bond Classification")
    parser.add_argument("--pdf-dir", type=str, default=PDF_DIR, help="Directory containing PDFs")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for ML dataset")
    
    args = parser.parse_args()
    
    process_all_pdfs(args.pdf_dir, args.output_dir)
