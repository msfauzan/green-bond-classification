"""
Canonical keyword lists for Green Bond Classification.
Based on POJK 18/2023 - Penerbitan dan Persyaratan Efek Bersifat Utang
dan/atau Sukuk Berlandaskan Keberlanjutan.

IMPORTANT: Only use SPECIFIC keywords that clearly identify the bond type.
Generic terms like "pendidikan", "kesehatan" appear in ALL prospectuses
(CSR sections) and should NOT be used as classification criteria.

NOTE: POJK 18/2023 is the GENERAL regulation for all EBUS types,
NOT specific to green bonds!
"""

# EBUS Lingkungan (Green Bond/Green Sukuk) - Pasal 1 ayat 3
# Dana hasil penerbitan digunakan untuk kegiatan usaha berwawasan LINGKUNGAN
GREEN_KEYWORDS = [
    # Primary identifiers - MUST be specific to green/environmental
    ('ebus lingkungan', 15), ('efek bersifat utang berwawasan lingkungan', 15),
    ('sukuk berwawasan lingkungan', 15), ('green bond', 12), ('green sukuk', 12),
    ('obligasi hijau', 12), ('sukuk hijau', 12), ('obligasi berwawasan lingkungan', 12),
    # KUBL - Kegiatan Usaha Berwawasan Lingkungan (Pasal 1 ayat 10)
    ('kubl', 10), ('kegiatan usaha berwawasan lingkungan', 10),
    # Green framework terms (specific to green bond prospectuses)
    ('green bond framework', 10), ('kerangka kerja obligasi hijau', 10),
    ('eligible green project', 8), ('proyek hijau yang memenuhi syarat', 8),
    ('penggunaan dana hasil penerbitan green', 8),
]

# EBUS Keberlanjutan (Sustainability Bond/Sukuk) - Pasal 1 ayat 5
# Dana hasil penerbitan digunakan untuk kegiatan LINGKUNGAN + SOSIAL
SUSTAINABILITY_KEYWORDS = [
    # Primary identifiers - specific to sustainability (kombinasi lingkungan + sosial)
    ('ebus keberlanjutan', 15), ('efek bersifat utang keberlanjutan', 15),
    ('sukuk keberlanjutan', 15), ('sustainability bond', 12), ('sustainability sukuk', 12),
    ('obligasi keberlanjutan', 12), ('obligasi berkelanjutan', 12),
    # Framework terms
    ('sustainability bond framework', 10), ('kerangka kerja obligasi keberlanjutan', 10),
    ('eligible sustainable project', 8),
    # EBUS Sosial components (Pasal 1 ayat 4) - jika ada sosial + lingkungan = sustainability
    ('ebus sosial', 10), ('efek bersifat utang berwawasan sosial', 10),
    ('sukuk berwawasan sosial', 10), ('social bond', 8), ('social sukuk', 8),
    ('obligasi sosial', 8),
    # KUBS - Kegiatan Usaha Berwawasan Sosial (Pasal 1 ayat 11)
    ('kubs', 10), ('kegiatan usaha berwawasan sosial', 10),
    ('eligible social project', 8), ('proyek sosial yang memenuhi syarat', 8),
]

# EBUS Terkait Keberlanjutan (Sustainability-Linked Bond/Sukuk) - Pasal 1 ayat 7
# Penerbitannya DIKAITKAN dengan pencapaian IKU dan TKK (bukan penggunaan dana)
SUSTAINABILITY_LINKED_KEYWORDS = [
    # Primary identifiers - MUST mention "terkait" or "linked"
    ('ebus terkait keberlanjutan', 15), ('efek bersifat utang terkait keberlanjutan', 15),
    ('sukuk terkait keberlanjutan', 15),
    ('sustainability linked bond', 12), ('sustainability-linked bond', 12),
    ('sustainability linked sukuk', 12), ('sustainability-linked sukuk', 12),
    ('obligasi terkait keberlanjutan', 12),
    # IKU Keberlanjutan - Indikator Kinerja Utama Keberlanjutan (Pasal 1 ayat 12)
    # This is THE KEY differentiator for SLB
    ('iku keberlanjutan', 12), ('indikator kinerja utama keberlanjutan', 12),
    ('sustainability key performance indicator', 10),
    # TKK - Target Kinerja Keberlanjutan (Pasal 1 ayat 13)
    ('target kinerja keberlanjutan', 12), ('tkk', 8),
    ('sustainability performance target', 10),
    # SLB Framework terms
    ('sustainability-linked bond framework', 10), ('slb framework', 8),
    # Mechanism characteristics unique to SLB (coupon adjustment based on KPI)
    ('step-up coupon', 8), ('step-down coupon', 8),
    ('coupon step-up', 8), ('coupon step-down', 8),
    ('penyesuaian tingkat bunga', 5),
]
