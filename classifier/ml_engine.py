"""
ML engine (Level 1) — semantic similarity via sentence-transformers.

Arsitektur:
  Level 0 (rules)       : SL / Wakaf (dari engine.py, tidak berubah)
  Title lookup (rules)  : cek nama obligasi di listing IDX → sinyal framing
                          paling andal (nama instrumen mencerminkan POJK 18/2023)
  Level 1 (ML)          : cosine-similarity antara segmen UoP dengan deskripsi
                          kategori taksonomi.
  Body framing (rules)  : fallback bila title tidak diketahui.

Tiga tier threshold berdasarkan sinyal framing:
  title_gss=True  → threshold 0.28  (yakin GSS, cukup konfirmasi semantik)
  title_gss=None  → threshold 0.30/0.52  (tak diketahui: andalkan body framing)
  title_gss=False → threshold 0.45/0.62  (bukan GSS, butuh bukti semantik kuat)

Model default: paraphrase-multilingual-MiniLM-L12-v2 (~120 MB, multilingual,
bebas, berjalan lokal). Diunduh otomatis dari HuggingFace pertama kali.

Explainability: setiap keputusan menyertakan skor per kategori + bukti framing.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .engine import (
    read_pdf_text, find_use_of_proceeds, detect_level0, Result,
)
from .taxonomy import (
    GSSClass, Bucket, ALL_CATEGORIES, BLUE_KEYWORDS, NEGATION_HINTS,
)
from .title_lookup import has_gss_title

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Tier threshold — makin tidak yakin GSS, makin tinggi bukti semantik yang dibutuhkan
THRESHOLD_TITLE_GSS    = 0.28   # title IDX konfirmasi GSS  → konfirmasi UoP
THRESHOLD_BODY_FRAMING = 0.30   # title tidak diketahui + body framing
THRESHOLD_NO_FRAMING   = 0.52   # title tidak diketahui + tanpa body framing
THRESHOLD_TITLE_NONGSS_FRAMING = 0.45  # title non-GSS tapi body ada framing (ragu)
THRESHOLD_TITLE_NONGSS = 0.62   # title non-GSS + tanpa body framing → sangat ketat

# ---------------------------------------------------------------------------
# Framing gate — sinyal komitmen GSS eksplisit
# ---------------------------------------------------------------------------
# Penting: hindari "berkelanjutan" sendirian (jebakan PUB — lihat CLAUDE.md)

FRAMING_SIGNALS: list[str] = [
    # Label POJK 18/2023 (paling andal)
    "berwawasan lingkungan", "berwawasan sosial", "berwawasan keberlanjutan",
    "berlandaskan keberlanjutan",
    # Label internasional
    "green bond", "green sukuk", "social bond", "blue bond",
    "sustainability bond", "sustainability sukuk",
    "obligasi hijau", "sukuk hijau", "obligasi sosial",
    # Framework & SPO
    "second party opinion", "green bond framework", "social bond framework",
    "sustainability bond framework", "framework keberlanjutan",
    "kerangka obligasi", "kerangka keberlanjutan penerbit",
    # Referensi regulasi
    "pojk 18/2023", "pojk no. 18", "pojk nomor 18", "pojk 18 tahun 2023",
    # Bahasa proyek eligible
    "eligible green", "eligible social", "eligible project",
    "proyek hijau yang memenuhi syarat", "proyek sosial yang memenuhi syarat",
    "kriteria kelayakan hijau", "kriteria kelayakan sosial",
    # Laporan alokasi & dampak
    "laporan alokasi dana", "laporan dampak lingkungan", "laporan dampak sosial",
    "allocation and impact report",
    # Prinsip ICMA
    "green bond principles", "social bond principles",
    "sustainability bond guidelines", "icma green", "icma social",
]


def has_framing_signal(text: str) -> tuple[bool, list[str]]:
    low = text.lower()
    hits = [s for s in FRAMING_SIGNALS if s in low]
    return bool(hits), hits


# ---------------------------------------------------------------------------
# Deskripsi kategori (sisi "query") — kalimat komitmen, bukan daftar kw
# ---------------------------------------------------------------------------
# Menggambarkan seperti apa PENGGUNAAN DANA dalam prospektus GSS sungguhan.

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "renewable_energy": (
        "Dana penawaran dialokasikan untuk membiayai atau merefinansiasi proyek "
        "pembangkit listrik energi terbarukan: tenaga surya, tenaga angin, panas bumi, "
        "tenaga air, biomassa, atau sumber energi baru terbarukan lainnya."
    ),
    "energy_efficiency": (
        "Dana dialokasikan untuk proyek efisiensi dan konservasi energi: retrofit gedung, "
        "smart grid, sistem manajemen energi, atau teknologi pengurangan konsumsi energi."
    ),
    "green_tourism": (
        "Dana dialokasikan untuk pengembangan pariwisata berkelanjutan dan ramah lingkungan "
        "serta ekowisata bersertifikat."
    ),
    "sustainable_transport": (
        "Dana dialokasikan untuk infrastruktur atau armada transportasi publik rendah emisi: "
        "kereta api listrik, MRT, LRT, bus rapid transit, kendaraan listrik."
    ),
    "green_building": (
        "Dana dialokasikan untuk konstruksi atau renovasi gedung bersertifikasi bangunan "
        "hijau: Greenship, EDGE, LEED, atau standar efisiensi energi yang diakui."
    ),
    "waste_management": (
        "Dana dialokasikan untuk pengelolaan sampah dan limbah: fasilitas pengolahan limbah "
        "industri, daur ulang, atau pembangkit listrik tenaga sampah."
    ),
    "water_management": (
        "Dana dialokasikan untuk infrastruktur air bersih dan sanitasi: sistem penyediaan "
        "air minum, instalasi pengolahan air limbah, irigasi berkelanjutan."
    ),
    "natural_resources": (
        "Dana dialokasikan untuk konservasi sumber daya alam berkelanjutan: kehutanan "
        "lestari, rehabilitasi lahan, pertanian berkelanjutan, konservasi keanekaragaman "
        "hayati, atau restorasi ekosistem."
    ),
    "climate_resilience": (
        "Dana dialokasikan untuk adaptasi dan ketahanan iklim: pengurangan risiko bencana, "
        "sistem peringatan dini, pengendalian banjir, atau adaptasi perubahan iklim."
    ),
    "basic_infrastructure": (
        "Dana dialokasikan sebagai obligasi sosial untuk infrastruktur dasar terjangkau "
        "bagi masyarakat miskin: elektrifikasi desa, akses air minum bersih, sanitasi dasar."
    ),
    "essential_services": (
        "Dana dialokasikan sebagai obligasi sosial untuk akses layanan kesehatan dan "
        "pendidikan bagi masyarakat kurang terlayani: puskesmas, rumah sakit, sekolah, "
        "beasiswa."
    ),
    "affordable_housing": (
        "Dana dialokasikan sebagai obligasi sosial untuk pembiayaan perumahan subsidi dan "
        "perumahan terjangkau bagi masyarakat berpenghasilan rendah."
    ),
    "employment_msme": (
        "Dana dialokasikan sebagai obligasi sosial secara khusus untuk memberdayakan "
        "usaha mikro kecil menengah termarjinalisasi dan menciptakan lapangan kerja bagi "
        "kelompok rentan, dengan target dampak sosial terukur."
    ),
    "food_security": (
        "Dana dialokasikan sebagai obligasi sosial untuk ketahanan pangan: pembiayaan "
        "petani kecil, infrastruktur pascapanen, distribusi pangan bagi masyarakat rentan."
    ),
    "socioeconomic": (
        "Dana dialokasikan sebagai obligasi sosial untuk pemberdayaan sosial-ekonomi "
        "masyarakat rentan, pengentasan kemiskinan, inklusi keuangan, atau bantuan sosial "
        "terstruktur."
    ),
}

# Sub-tema Blue (kw sederhana, cukup untuk tag tambahan)
_BLUE_LOW = [k.lower() for k in BLUE_KEYWORDS]


# ---------------------------------------------------------------------------
# Model (lazy-load, singleton)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def _category_embeddings(model_name: str = DEFAULT_MODEL) -> dict[str, np.ndarray]:
    model = _get_model(model_name)
    return {
        key: model.encode(desc, normalize_embeddings=True)
        for key, desc in CATEGORY_DESCRIPTIONS.items()
    }


# ---------------------------------------------------------------------------
# Klasifikasi ML
# ---------------------------------------------------------------------------

@dataclass
class MLResult:
    gss_class: GSSClass
    scores: dict[str, float]            # similarity per kategori
    top_env: list[tuple[str, float]]    # (key, score) kategori lingkungan
    top_soc: list[tuple[str, float]]    # (key, score) kategori sosial
    blue: list[str]
    framing_body: list[str]             # sinyal framing dari badan dok
    framing_title: list[str]            # nama obligasi GSS dari listing IDX
    title_gss: bool | None             # True/False/None (tidak diketahui)
    level0_evidence: list[str]
    anchor: str
    threshold_used: float
    confidence: float

    @property
    def is_gss(self) -> bool:
        return self.gss_class != GSSClass.NON_GSS

    def sector_keys(self) -> list[str]:
        return [k for k, _ in self.top_env] + [k for k, _ in self.top_soc]


def classify_ml(
    text: str,
    issuer: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> MLResult | None:
    if not text or not text.strip():
        return None

    # Level 0 (rules) — deteksi struktur instrumen
    lvl0, l0ev = detect_level0(text)

    # --- Sinyal framing (dua sumber) ---
    # Tier 1: nama obligasi di listing IDX (paling andal)
    title_gss, title_hits = has_gss_title(issuer) if issuer else (None, [])
    # Tier 2: bahasa komitmen GSS di badan dokumen (fallback)
    body_framed, body_hits = has_framing_signal(text)

    # Pilih threshold sesuai tier
    if title_gss is True:
        threshold = THRESHOLD_TITLE_GSS
    elif title_gss is False:
        threshold = THRESHOLD_TITLE_NONGSS_FRAMING if body_framed else THRESHOLD_TITLE_NONGSS
    else:  # title_gss is None (tidak ada di listing)
        threshold = THRESHOLD_BODY_FRAMING if body_framed else THRESHOLD_NO_FRAMING

    # Extract UoP segment
    seg, anchor = find_use_of_proceeds(text)
    seg_low = seg.lower()
    blue = [k for k in _BLUE_LOW if k in seg_low]

    if lvl0 is not None:
        return MLResult(
            gss_class=lvl0, scores={},
            top_env=[], top_soc=[], blue=blue,
            framing_body=body_hits, framing_title=title_hits,
            title_gss=title_gss, level0_evidence=l0ev,
            anchor=anchor, threshold_used=threshold, confidence=0.90,
        )

    # --- Level 1: semantic similarity ---
    model   = _get_model(model_name)
    seg_emb = model.encode(seg, normalize_embeddings=True)
    cat_embs = _category_embeddings(model_name)

    cat_map = {c.key: c for c in ALL_CATEGORIES}
    scores: dict[str, float] = {
        key: float(np.dot(seg_emb, emb)) for key, emb in cat_embs.items()
    }

    env_hits = [
        (k, s) for k, s in scores.items()
        if s >= threshold and cat_map[k].bucket.value == "environmental"
    ]
    soc_hits = [
        (k, s) for k, s in scores.items()
        if s >= threshold and cat_map[k].bucket.value == "social"
    ]
    env_hits.sort(key=lambda x: -x[1])
    soc_hits.sort(key=lambda x: -x[1])

    if env_hits and soc_hits:
        gss = GSSClass.SUSTAINABILITY
    elif env_hits:
        gss = GSSClass.GREEN
    elif soc_hits:
        gss = GSSClass.SOCIAL
    else:
        gss = GSSClass.NON_GSS

    max_score = max(scores.values()) if scores else 0.0
    title_boost = 0.12 if title_gss is True else (0.05 if body_framed else 0.0)
    if gss == GSSClass.NON_GSS:
        conf = round(min(0.88, 0.40 + (threshold - max_score) * 2), 2)
    else:
        conf = round(min(0.96, 0.50 + (max_score - threshold) * 1.5 + title_boost), 2)

    return MLResult(
        gss_class=gss, scores=scores,
        top_env=env_hits, top_soc=soc_hits, blue=blue,
        framing_body=body_hits, framing_title=title_hits,
        title_gss=title_gss, level0_evidence=l0ev,
        anchor=anchor, threshold_used=threshold, confidence=conf,
    )
