"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GENERATE LABELED DATASET (Pseudo-Labels)                            ║
║           Bank Indonesia - DSta-DSMF - Green Bond Classification             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import PyPDF2
import pandas as pd
from datetime import datetime

BASE_DIR = r'd:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification'
OUTPUT_DIR = os.path.join(BASE_DIR, 'ML_Dataset')

# Folders containing PDFs
PDF_FOLDERS = [
    os.path.join(BASE_DIR, 'Prospektus_Downloaded'),
    os.path.join(BASE_DIR, 'Prospektus Obligasi'),
    os.path.join(BASE_DIR, 'test'),
]

GREEN_KEYWORDS = [
    ('green bond', 10), ('green sukuk', 10), ('obligasi hijau', 10),
    ('kubl', 8), ('pojk 60', 5), ('proyek hijau', 5),
    ('energi terbarukan', 2), ('renewable energy', 2),
]
SUSTAINABILITY_KEYWORDS = [
    ('sustainability bond', 10), ('sustainability sukuk', 10),
    ('obligasi keberlanjutan', 10), ('ebus keberlanjutan', 10),
    ('proyek sosial', 5), ('sdgs', 2),
]
SUSTAINABILITY_LINKED_KEYWORDS = [
    ('sustainability linked bond', 10), ('sustainability-linked bond', 10),
    ('sustainability performance target', 8), ('kpi keberlanjutan', 8),
    ('step-up', 3), ('step-down', 3),
]

def calc_score(text, keywords):
    text_lower = text.lower()
    total = 0
    found = []
    for kw, w in keywords:
        c = text_lower.count(kw.lower())
        if c > 0:
            total += w * min(c, 5)
            found.append(f'{kw}({c})')
    return total, '; '.join(found)

def classify(g, s, l):
    if l >= 10 and l >= s and l >= g: return 'sustainability_linked_bond'
    elif s >= 10 and s >= g: return 'sustainability_bond'
    elif g >= 10: return 'green_bond'
    else: return 'obligasi_biasa'

def get_pdfs(folders):
    pdfs = []
    for f in folders:
        if os.path.exists(f):
            for root, dirs, files in os.walk(f):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdfs.append(os.path.join(root, file))
    return pdfs

def main():
    print("="*70)
    print("GENERATE LABELED DATASET")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_pdfs = get_pdfs(PDF_FOLDERS)
    print(f'\n📂 Found {len(all_pdfs)} PDFs\n')

    results = []
    for i, path in enumerate(all_pdfs, 1):
        fn = os.path.basename(path)
        print(f'[{i}/{len(all_pdfs)}] {fn[:50]}...', end=' ')
        try:
            with open(path, 'rb') as f:
                r = PyPDF2.PdfReader(f)
                text = ' '.join([(p.extract_text() or '') for p in r.pages])
                pages = len(r.pages)
            
            g, gk = calc_score(text, GREEN_KEYWORDS)
            s, sk = calc_score(text, SUSTAINABILITY_KEYWORDS)
            l, lk = calc_score(text, SUSTAINABILITY_LINKED_KEYWORDS)
            label = classify(g, s, l)
            
            conf = 0.9 if max(g,s,l) >= 30 else (0.7 if max(g,s,l) >= 10 else 0.5)
            
            results.append({
                'filename': fn, 'filepath': path, 'pages': pages,
                'words': len(text.split()),
                'green_score': g, 'sustain_score': s, 'linked_score': l,
                'green_kw': gk, 'sustain_kw': sk, 'linked_kw': lk,
                'label': label, 'confidence': conf,
                'manual_label': '',
                'is_verified': False
            })
            emoji = {'green_bond':'🌿','sustainability_bond':'♻️','sustainability_linked_bond':'🔗','obligasi_biasa':'📄'}
            print(f'{emoji.get(label,"?")} {label}')
        except Exception as e:
            print(f'Error: {e}')

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'labeled_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n✅ Saved {len(df)} records to {csv_path}')

    print('\n📊 Label Distribution:')
    for lbl, cnt in df['label'].value_counts().items():
        print(f'  {lbl}: {cnt}')
    
    return df

if __name__ == "__main__":
    main()
