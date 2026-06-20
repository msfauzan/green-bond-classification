"""
ML engine (Level 1) — semantic similarity via sentence-transformers.

Arsitektur:
  Level 0 (rules)   : SL / Wakaf (dari engine.py, tidak berubah)
  Level 1 (ML)      : cosine-similarity antara segmen UoP dengan deskripsi
                      kategori taksonomi yang diembed.
  Framing gate (rule): haruskan sinyal komitmen GSS eksplisit; tanpanya,
                      butuh threshold semantik lebih tinggi.
                      → Fix utama untuk employment_msme FP.

Model default: paraphrase-multilingual-MiniLM-L12-v2 (~120 MB, multilingual,
bebas, berjalan lokal). Diunduh otomatis dari HuggingFace pertama kali.

Explainability: setiap keputusan menyertakan skor per kategori.
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

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Tanpa framing signal, butuh threshold lebih tinggi agar tidak FP
THRESHOLD_WITH_FRAMING    = 0.30
THRESHOLD_WITHOUT_FRAMING = 0.52

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
    framing: list[str]
    level0_evidence: list[str]
    anchor: str
    threshold_used: float
    confidence: float

    @property
    def is_gss(self) -> bool:
        return self.gss_class != GSSClass.NON_GSS

    def sector_keys(self) -> list[str]:
        return [k for k, _ in self.top_env] + [k for k, _ in self.top_soc]


def classify_ml(text: str, model_name: str = DEFAULT_MODEL) -> MLResult | None:
    if not text or not text.strip():
        return None

    # Level 0 (rules)
    lvl0, l0ev = detect_level0(text)

    # Framing gate
    framed, framing_hits = has_framing_signal(text)
    threshold = THRESHOLD_WITH_FRAMING if framed else THRESHOLD_WITHOUT_FRAMING

    # Extract UoP segment
    seg, anchor = find_use_of_proceeds(text)

    # Sub-tema Blue (rule — cukup untuk tag)
    seg_low = seg.lower()
    blue = [k for k in _BLUE_LOW if k in seg_low]

    if lvl0 is not None:
        scores = {}
        return MLResult(
            gss_class=lvl0, scores=scores,
            top_env=[], top_soc=[], blue=blue,
            framing=framing_hits, level0_evidence=l0ev,
            anchor=anchor, threshold_used=threshold, confidence=0.90,
        )

    # Encode UoP segment (model truncates panjang otomatis)
    model = _get_model(model_name)
    seg_emb = model.encode(seg, normalize_embeddings=True)
    cat_embs = _category_embeddings(model_name)

    cat_map = {c.key: c for c in ALL_CATEGORIES}
    scores: dict[str, float] = {
        key: float(np.dot(seg_emb, emb)) for key, emb in cat_embs.items()
    }

    # Pisahkan env vs social berdasarkan bucket
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

    # Confidence: maks skor di atas threshold, ditambah boost framing
    max_score = max(scores.values()) if scores else 0.0
    framing_boost = 0.10 if framed else 0.0
    if gss == GSSClass.NON_GSS:
        conf = round(min(0.85, 0.40 + (threshold - max_score) * 2 + framing_boost), 2)
    else:
        conf = round(min(0.95, 0.50 + (max_score - threshold) * 1.5 + framing_boost), 2)

    return MLResult(
        gss_class=gss, scores=scores,
        top_env=env_hits, top_soc=soc_hits, blue=blue,
        framing=framing_hits, level0_evidence=l0ev,
        anchor=anchor, threshold_used=threshold, confidence=conf,
    )
