"""
Taksonomi acuan untuk klasifikasi EBUS GSS — "kamus kebenaran".

Definisi green/social TIDAK dipelajari dari data, melainkan di-encode dari
regulasi/standar resmi:
  - Kategori Green & Blue eligible  : SDG Government Securities Framework (DJPPR)
  - Prinsip & kategori                : ICMA Green/Social Bond Principles
  - Kerangka nasional                 : POJK 18/2023

Skema klasifikasi BERTINGKAT (lihat Kerangka_AN):
  Level 0 (struktur instrumen):
      - Sustainability-Linked  -> dideteksi via sinyal KPI/SPT/step-up
      - Sukuk Wakaf            -> dideteksi via klausa wakaf
      - else                   -> Use-of-Proceeds  (lanjut Level 1)
  Level 1 (hanya untuk Use-of-Proceeds), berdasarkan kategori proyek:
      - kategori lingkungan saja        -> GREEN
      - kategori sosial saja            -> SOCIAL
      - campuran lingkungan + sosial    -> SUSTAINABILITY
      - kata kunci kelautan             -> tambah sub-tema BLUE

Catatan: daftar kata kunci di bawah adalah TITIK MULAI yang harus
disempurnakan & ditinjau berkala (sinonim, istilah baru). Bilingual ID/EN
karena dokumen pasar Indonesia bercampur dua bahasa.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Kelas keluaran
# ---------------------------------------------------------------------------

class GSSClass(str, Enum):
    GREEN = "green"
    SOCIAL = "social"
    SUSTAINABILITY = "sustainability"
    SUSTAINABILITY_LINKED = "sustainability_linked"
    WAKAF = "sukuk_wakaf"
    NON_GSS = "obligasi_biasa"          # tidak terindikasi GSS


class Bucket(str, Enum):
    """Pengelompokan kategori use-of-proceeds ke arah GSS-nya."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"


# ---------------------------------------------------------------------------
# Level 0 — sinyal STRUKTUR instrumen (bukan soal penggunaan dana)
# ---------------------------------------------------------------------------
# Sustainability-Linked: dana bebas; yang menentukan adalah mekanisme target
# kinerja. Sinyal ini KUAT & spesifik -> precision tinggi.
SUSTAINABILITY_LINKED_SIGNALS: list[str] = [
    "sustainability-linked", "sustainability linked", "terkait keberlanjutan",
    "key performance indicator", "indikator kinerja utama", "kpi",
    "sustainability performance target", "target kinerja keberlanjutan", "spt",
    "step-up", "step up", "penyesuaian tingkat bunga", "penyesuaian kupon",
    "margin ratchet", "coupon step-up", "kenaikan kupon",
]

# Sukuk Wakaf: instrumen wakaf (CWLS dsb.)
WAKAF_SIGNALS: list[str] = [
    "wakaf", "waqf", "sukuk wakaf", "cash waqf", "cwls", "nazhir", "nadzir",
    "ikrar wakaf", "mauquf",
]


# ---------------------------------------------------------------------------
# Level 1 — kategori USE-OF-PROCEEDS (eligible)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Category:
    key: str
    name_id: str
    bucket: Bucket
    keywords: tuple[str, ...]
    blue: bool = False           # True jila kategori bertema kelautan
    weight: float = 1.0          # bobot relatif (bisa di-tune ke gold set)


# --- Kategori Lingkungan (Green & Blue) — 9 sektor DJPPR ---
GREEN_CATEGORIES: list[Category] = [
    Category("renewable_energy", "Energi Terbarukan", Bucket.ENVIRONMENTAL, (
        "energi terbarukan", "energi baru terbarukan", "ebt", "renewable energy",
        "panel surya", "tenaga surya", "plts", "solar pv", "fotovoltaik",
        "tenaga angin", "pltb", "wind power", "panas bumi", "geotermal", "pltp",
        "geothermal", "tenaga air", "plta", "hydropower", "mikrohidro",
        "biomassa", "biomass", "biogas", "bioenergi",
    )),
    Category("energy_efficiency", "Efisiensi Energi", Bucket.ENVIRONMENTAL, (
        "efisiensi energi", "energy efficiency", "konservasi energi",
        "hemat energi", "smart grid", "jaringan pintar", "retrofit",
        "manajemen energi", "kogenerasi", "cogeneration",
    )),
    Category("green_tourism", "Pariwisata Berkelanjutan", Bucket.ENVIRONMENTAL, (
        "pariwisata berkelanjutan", "ekowisata", "green tourism",
        "sustainable tourism", "wisata ramah lingkungan",
    )),
    Category("sustainable_transport", "Transportasi Berkelanjutan", Bucket.ENVIRONMENTAL, (
        "transportasi berkelanjutan", "transportasi ramah lingkungan",
        "sustainable transport", "clean transportation", "transportasi massal",
        "angkutan massal", "mrt", "lrt", "krl", "bus rapid transit", "brt",
        "transjakarta", "kereta api", "kendaraan listrik", "electric vehicle",
        "kendaraan rendah emisi", "stasiun pengisian",
    )),
    Category("green_building", "Bangunan Hijau", Bucket.ENVIRONMENTAL, (
        "bangunan hijau", "gedung hijau", "green building", "greenship",
        "edge certification", "sertifikasi bangunan hijau", "green property",
    )),
    Category("waste_management", "Pengelolaan Sampah & Limbah", Bucket.ENVIRONMENTAL, (
        "pengelolaan sampah", "pengelolaan limbah", "waste management",
        "waste to energy", "pltsa", "pembangkit listrik tenaga sampah",
        "daur ulang", "recycling", "pengolahan limbah", "3r",
    )),
    Category("water_management", "Pengelolaan Air & Air Limbah", Bucket.ENVIRONMENTAL, (
        "pengelolaan air", "air bersih", "sustainable water", "spam",
        "sistem penyediaan air minum", "ipal", "instalasi pengolahan air limbah",
        "pengolahan air limbah", "wastewater", "sanitasi", "irigasi",
    )),
    Category("natural_resources", "Pengelolaan SDA Berkelanjutan (Darat & Laut)", Bucket.ENVIRONMENTAL, (
        "pengelolaan sumber daya alam", "kehutanan lestari",
        "sustainable forestry", "perhutanan sosial", "reboisasi",
        "rehabilitasi lahan", "konservasi", "biodiversitas", "keanekaragaman hayati",
        "pertanian berkelanjutan", "sustainable agriculture",
        # sub-tema kelautan (Blue) ditandai terpisah di BLUE_KEYWORDS
        "perikanan berkelanjutan", "akuakultur berkelanjutan",
    ), blue=False),
    Category("climate_resilience", "Ketahanan Iklim & Pengurangan Risiko Bencana", Bucket.ENVIRONMENTAL, (
        "ketahanan iklim", "climate resilience", "adaptasi perubahan iklim",
        "climate adaptation", "pengurangan risiko bencana",
        "disaster risk reduction", "mitigasi bencana", "pengendalian banjir",
        "early warning", "sistem peringatan dini",
    )),
]

# --- Kategori Sosial — ICMA Social Bond Principles / POJK ---
SOCIAL_CATEGORIES: list[Category] = [
    Category("basic_infrastructure", "Infrastruktur Dasar Terjangkau", Bucket.SOCIAL, (
        "infrastruktur dasar", "affordable basic infrastructure",
        "akses air minum", "sanitasi dasar", "listrik perdesaan", "elektrifikasi",
    )),
    Category("essential_services", "Akses Layanan Esensial", Bucket.SOCIAL, (
        "akses kesehatan", "layanan kesehatan", "rumah sakit", "puskesmas",
        "akses pendidikan", "pendidikan", "beasiswa", "sekolah", "fasilitas kesehatan",
        "healthcare", "education",
    )),
    Category("affordable_housing", "Perumahan Terjangkau", Bucket.SOCIAL, (
        "perumahan terjangkau", "affordable housing", "rumah subsidi",
        "rumah rakyat", "masyarakat berpenghasilan rendah", "mbr", "rumah susun",
    )),
    Category("employment_msme", "Penciptaan Lapangan Kerja & UMKM", Bucket.SOCIAL, (
        "penciptaan lapangan kerja", "employment generation", "umkm", "umk",
        "usaha mikro", "pemberdayaan umkm", "kredit usaha rakyat", "kur",
        "msme", "pembiayaan mikro", "microfinance",
    )),
    Category("food_security", "Ketahanan Pangan", Bucket.SOCIAL, (
        "ketahanan pangan", "food security", "kedaulatan pangan",
        "produktivitas pertanian", "pasca panen",
    )),
    Category("socioeconomic", "Pemberdayaan Sosial-Ekonomi", Bucket.SOCIAL, (
        "pemberdayaan", "pengentasan kemiskinan", "poverty alleviation",
        "program keluarga harapan", "pkh", "bantuan sosial", "bansos",
        "socioeconomic advancement", "inklusi keuangan", "financial inclusion",
    )),
]

ALL_CATEGORIES: list[Category] = GREEN_CATEGORIES + SOCIAL_CATEGORIES

# --- Sub-tema BLUE (kelautan) — tag tambahan, bukan kelas tersendiri ---
BLUE_KEYWORDS: list[str] = [
    "kelautan", "maritim", "perikanan", "akuakultur", "mangrove", "pesisir",
    "terumbu karang", "konservasi laut", "ekosistem laut", "blue economy",
    "ekonomi biru", "blue bond", "blue financing", "marine", "coastal",
    "ocean", "fisheries",
]

# --- Petunjuk NEGASI / pengecualian (untuk menekan false positive) ---
# Frasa yang menandakan kata kunci muncul dalam konteks MENYANGKAL/umum.
NEGATION_HINTS: list[str] = [
    "tidak termasuk", "bukan merupakan", "selain", "tidak digunakan untuk",
    "dikecualikan", "tanpa", "not include", "excluding",
]


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def category_index() -> dict[str, Category]:
    """Map key -> Category."""
    return {c.key: c for c in ALL_CATEGORIES}


def keyword_to_category() -> dict[str, Category]:
    """Map setiap keyword -> Category pemiliknya (untuk audit/penjelasan)."""
    out: dict[str, Category] = {}
    for c in ALL_CATEGORIES:
        for kw in c.keywords:
            out[kw] = c
    return out
