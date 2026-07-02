"""
Regression check untuk bug substring keyword (lihat CLAUDE.md / commit terkait):
"kur"/"umk" (employment_msme) dulu cocok di tengah kata tak-terkait
("dikurangi", "dicantumkan") dan membuat prospektus Green Bond murni
terklasifikasi SUSTAINABILITY secara keliru.

Jalankan (dari root repo): python -m classifier.test_engine
"""
from __future__ import annotations
from classifier.taxonomy import GSSClass
from classifier.engine import match_categories, classify

# --- 1. "kur"/"umk" tidak boleh cocok di tengah kata tak-terkait ---
noise = (
    "Dana yang diperoleh setelah dikurangi biaya emisi akan dicantumkan "
    "dalam laporan tahunan, sekurang-kurangnya sesuai jadwal."
)
hits = {c.key for c, _ in match_categories(noise)}
assert "employment_msme" not in hits, f"false positive: {hits}"

# --- 2. tapi tetap terdeteksi bila memang kata utuh ---
real = "Dana dialokasikan untuk pembiayaan UMKM melalui skema KUR."
hits2 = {c.key for c, _ in match_categories(real)}
assert "employment_msme" in hits2, f"kata utuh KUR/UMKM harus tetap terdeteksi: {hits2}"

# --- 3. prospektus Green Bond murni (tanpa proyek sosial) -> GREEN, bukan SUSTAINABILITY ---
green_only_uop = (
    "RENCANA PENGGUNAAN DANA\n"
    "Dana yang diperoleh dari Penawaran Umum Green Bond setelah dikurangi biaya-biaya "
    "emisi akan digunakan untuk proyek energi terbarukan, efisiensi energi, dan "
    "pengelolaan air, sebagaimana dicantumkan dalam Kerangka Green Bond, sekurang-kurangnya "
    "sesuai POJK 60/POJK.04/2017."
)
result = classify(green_only_uop)
assert result.gss_class == GSSClass.GREEN, f"expected GREEN, got {result.gss_class}"
assert not result.soc_sectors, f"tidak boleh ada sektor sosial: {result.soc_sectors}"

print("OK - semua regression check lulus")
