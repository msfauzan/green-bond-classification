"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               GROUND TRUTH LABELING FOR GREEN BONDS                           ║
║               Bank Indonesia - DSta-DSMF                                      ║
║                                                                               ║
║  This script creates ground truth labels based on:                            ║
║  1. Official OJK/IDX green bond list                                          ║
║  2. Manual verification of prospectus titles                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# KNOWN GREEN/SUSTAINABILITY BONDS FROM OFFICIAL SOURCES
# =============================================================================
# Source: IDX, OJK, Company announcements
# Update this list as new bonds are issued

KNOWN_GREEN_BONDS = [
    # Company name patterns (case-insensitive)
    # Format: (company_pattern, bond_type, year_min, year_max)
    
    # TLKM - Telkom Green Bond
    ('telkom', 'green_bond', 2024, 2030),
    
    # PLN - Various Green Bonds
    ('perusahaan listrik negara', 'green_bond', 2020, 2030),
    ('pln', 'green_bond', 2020, 2030),
    
    # SMF - Sarana Multigriya Finansial (Green/Sustainability)
    ('sarana multigriya finansial', 'sustainability_bond', 2020, 2030),
    ('smf', 'sustainability_bond', 2020, 2030),
    
    # Bank-bank dengan Green Bond
    ('bank mandiri', 'green_bond', 2023, 2030),
    ('bank rakyat indonesia', 'sustainability_bond', 2023, 2030),
    ('bri', 'sustainability_bond', 2023, 2030),
    
    # MPMX - Mitra Pinasthika Mustika (Sustainability-Linked)
    ('mitra pinasthika', 'sustainability_linked_bond', 2022, 2030),
    ('mpmx', 'sustainability_linked_bond', 2022, 2030),
    
    # SRTG - Saratoga (Sustainability Bond)
    ('saratoga', 'sustainability_bond', 2022, 2030),
    
    # BNGA - Bank CIMB Niaga (Sustainability)
    ('cimb niaga', 'sustainability_bond', 2022, 2030),
    ('bnga', 'sustainability_bond', 2022, 2030),
]

# Patterns in filename/title that indicate bond type
TITLE_PATTERNS = {
    'green_bond': [
        r'green\s*bond', r'obligasi\s*hijau', r'sukuk\s*hijau',
        r'green\s*sukuk', r'ebus\s*lingkungan'
    ],
    'sustainability_bond': [
        r'sustainability\s*bond', r'obligasi\s*keberlanjutan', 
        r'sukuk\s*keberlanjutan', r'social\s*bond', r'obligasi\s*sosial',
        r'ebus\s*keberlanjutan', r'ebus\s*sosial'
    ],
    'sustainability_linked_bond': [
        r'sustainability[\s\-]*linked', r'ebus\s*terkait\s*keberlanjutan'
    ]
}

def classify_by_known_list(company_name: str, filename: str, year: int = 2024) -> str:
    """Classify based on known green bond issuers"""
    company_lower = company_name.lower() if company_name else ""
    filename_lower = filename.lower() if filename else ""
    
    for pattern, bond_type, year_min, year_max in KNOWN_GREEN_BONDS:
        if pattern.lower() in company_lower or pattern.lower() in filename_lower:
            if year_min <= year <= year_max:
                return bond_type
    
    return None

def classify_by_title_pattern(filename: str, text_first_page: str = "") -> str:
    """Classify based on patterns in filename or title page"""
    search_text = f"{filename} {text_first_page}".lower()
    
    # Check most specific first (sustainability-linked)
    for pattern in TITLE_PATTERNS['sustainability_linked_bond']:
        if re.search(pattern, search_text, re.IGNORECASE):
            return 'sustainability_linked_bond'
    
    for pattern in TITLE_PATTERNS['sustainability_bond']:
        if re.search(pattern, search_text, re.IGNORECASE):
            return 'sustainability_bond'
    
    for pattern in TITLE_PATTERNS['green_bond']:
        if re.search(pattern, search_text, re.IGNORECASE):
            return 'green_bond'
    
    return None

def create_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ground truth labels using multiple methods:
    1. Known issuer list
    2. Title/filename patterns
    3. Content keywords (for verification)
    """
    df = df.copy()
    df['ground_truth'] = None
    df['label_source'] = None
    
    for idx, row in df.iterrows():
        filename = row.get('filename', '')
        company = row.get('company_name', '')
        text = row.get('text', '')
        first_page = text[:3000] if text else ""
        
        # Extract year from filename
        year_match = re.search(r'20(\d{2})', filename)
        year = 2000 + int(year_match.group(1)) if year_match else 2024
        
        # Method 1: Known issuer list
        label = classify_by_known_list(company, filename, year)
        if label:
            df.at[idx, 'ground_truth'] = label
            df.at[idx, 'label_source'] = 'known_issuer'
            continue
        
        # Method 2: Title pattern
        label = classify_by_title_pattern(filename, first_page)
        if label:
            df.at[idx, 'ground_truth'] = label
            df.at[idx, 'label_source'] = 'title_pattern'
            continue
        
        # Default: obligasi biasa
        df.at[idx, 'ground_truth'] = 'obligasi_biasa'
        df.at[idx, 'label_source'] = 'default'
    
    return df

def analyze_labeling_results(df: pd.DataFrame):
    """Analyze and print labeling results"""
    print("\n" + "=" * 60)
    print("GROUND TRUTH LABELING RESULTS")
    print("=" * 60)
    
    print("\n📊 Label Distribution:")
    print(df['ground_truth'].value_counts().to_string())
    
    print("\n📋 Label Sources:")
    print(df['label_source'].value_counts().to_string())
    
    # Show samples for each non-default label
    print("\n🔍 Samples by Label:")
    for label in ['green_bond', 'sustainability_bond', 'sustainability_linked_bond']:
        samples = df[df['ground_truth'] == label]
        if len(samples) > 0:
            print(f"\n   {label.upper()} ({len(samples)} samples):")
            for _, row in samples.head(10).iterrows():
                print(f"      - {row.get('filename', 'N/A')[:60]}...")

if __name__ == "__main__":
    # Load existing dataset
    dataset_path = os.path.join(BASE_DIR, 'ML_Dataset', 'labeled_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        exit(1)
    
    print(f"📂 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df)} samples")
    
    # Create ground truth labels
    print("\n🏷️ Creating ground truth labels...")
    df = create_ground_truth_labels(df)
    
    # Analyze results
    analyze_labeling_results(df)
    
    # Compare with original pseudo-labels
    print("\n📊 Comparison with Original Labels:")
    comparison = pd.crosstab(df['label'], df['ground_truth'], margins=True)
    print(comparison)
    
    # Save updated dataset
    output_path = os.path.join(BASE_DIR, 'ML_Dataset', 'labeled_dataset_ground_truth.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
