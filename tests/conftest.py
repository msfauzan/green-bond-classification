"""Shared test fixtures."""
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def green_bond_text():
    """Sample text that should classify as green bond."""
    return (
        "PROSPEKTUS Green Bond Framework PT Contoh Hijau Tbk "
        "Obligasi hijau ini diterbitkan sesuai POJK 18/2023. "
        "EBUS Lingkungan efek bersifat utang berwawasan lingkungan. "
        "Kegiatan usaha berwawasan lingkungan KUBL eligible green project. "
        "Green bond framework kerangka kerja obligasi hijau. "
        "Penggunaan dana hasil penerbitan green untuk proyek hijau yang memenuhi syarat."
    )


@pytest.fixture
def sustainability_bond_text():
    """Sample text that should classify as sustainability bond."""
    return (
        "PROSPEKTUS Sustainability Bond Framework PT Contoh Berkelanjutan Tbk "
        "Obligasi keberlanjutan ini diterbitkan sesuai POJK 18/2023. "
        "EBUS Keberlanjutan efek bersifat utang keberlanjutan. "
        "Sustainability bond framework kerangka kerja obligasi keberlanjutan. "
        "EBUS Sosial efek bersifat utang berwawasan sosial kegiatan usaha berwawasan sosial KUBS. "
        "Eligible social project proyek sosial yang memenuhi syarat."
    )


@pytest.fixture
def sustainability_linked_text():
    """Sample text that should classify as sustainability-linked bond."""
    return (
        "PROSPEKTUS Sustainability-Linked Bond Framework PT Contoh Terkait Tbk "
        "Obligasi terkait keberlanjutan ini diterbitkan sesuai POJK 18/2023. "
        "EBUS Terkait Keberlanjutan efek bersifat utang terkait keberlanjutan. "
        "IKU Keberlanjutan indikator kinerja utama keberlanjutan. "
        "Target kinerja keberlanjutan sustainability performance target TKK. "
        "Step-up coupon step-down coupon penyesuaian tingkat bunga. "
        "SLB framework sustainability-linked bond framework."
    )


@pytest.fixture
def regular_bond_text():
    """Sample text that should classify as obligasi biasa."""
    return (
        "PROSPEKTUS Obligasi PT Contoh Biasa Tbk "
        "Obligasi ini diterbitkan dalam rangka pendanaan umum perusahaan. "
        "Perseroan bergerak di bidang manufaktur dan perdagangan. "
        "Dana hasil penerbitan obligasi akan digunakan untuk modal kerja "
        "dan pembayaran utang perseroan."
    )
