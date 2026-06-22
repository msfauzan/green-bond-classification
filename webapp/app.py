"""
Web app (Streamlit) — Klasifikasi EBUS GSS, Bank Indonesia (DSta/DSMF).

Tiga tab:
  1. Klasifikasi Prospektus  — unggah PDF / tempel teks → kelas GSS + bukti
  2. Statistik Pasar GSS      — scan title-lookup ke semesta 1.437 EBUS IDX
  3. Evaluasi Model           — metrik & hasil per-dokumen (gold set)

Jalankan dari root repo:
    streamlit run webapp/app.py

Semua lokal & bebas (sentence-transformers + PyMuPDF). Tanpa API berbayar.
"""
from __future__ import annotations

import os
import sys
import csv

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classifier.engine import read_pdf_bytes, find_use_of_proceeds
from classifier.ml_engine import classify_ml
from classifier.taxonomy import GSSClass, category_index
from classifier.title_lookup import (
    all_instruments, gss_type_from_title, has_gss_title, _load_idx,
)

DATA = os.path.join(ROOT, "data")
CAT = category_index()

# Tampilan kelas: label + emoji
CLASS_VIEW = {
    GSSClass.GREEN.value:                 ("🟢 Green — Lingkungan", "#16a34a"),
    GSSClass.SOCIAL.value:                ("🟠 Social — Sosial", "#ea580c"),
    GSSClass.SUSTAINABILITY.value:        ("🔵 Sustainability — Keberlanjutan", "#2563eb"),
    GSSClass.SUSTAINABILITY_LINKED.value: ("🟣 Sustainability-Linked", "#9333ea"),
    GSSClass.WAKAF.value:                 ("🟤 Sukuk Wakaf", "#92400e"),
    GSSClass.NON_GSS.value:               ("⚪ Non-GSS — Obligasi Biasa", "#6b7280"),
}

TYPE_LABEL = {
    "green": "Green", "social": "Social", "sustainability": "Sustainability",
    "sustainability_linked": "Sustainability-Linked", "blue": "Blue",
}

st.set_page_config(page_title="Klasifikasi EBUS GSS — BI", page_icon="🌱", layout="wide")


# ---------------------------------------------------------------------------
# Cache: model & data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Memuat model semantik (sekali saja)…")
def _warm_model():
    from classifier.ml_engine import _get_model, _category_embeddings
    _get_model()
    _category_embeddings()
    return True


@st.cache_data
def load_universe() -> pd.DataFrame:
    rows = all_instruments()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["gss_type"] = df["BondName"].map(gss_type_from_title)
    df["is_gss"] = df["gss_type"].notna()
    df["Outstanding"] = pd.to_numeric(df.get("Outstanding"), errors="coerce").fillna(0)
    return df


@st.cache_data
def load_comparison() -> pd.DataFrame:
    p = os.path.join(DATA, "comparison_results.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


@st.cache_data
def issuer_list() -> list[str]:
    return sorted(_load_idx().keys())


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🌱 Klasifikasi EBUS GSS")
st.caption(
    "Bank Indonesia · DSta / DSMF — klasifikasi Efek Bersifat Utang & Sukuk ke "
    "kategori Green / Social / Sustainability (POJK 18/2023, ICMA, DJPPR). "
    "Pipeline taxonomy-grounded: *rule-based* + *ML semantic* lokal."
)

tab_clf, tab_market, tab_eval = st.tabs(
    ["🔎 Klasifikasi Prospektus", "📊 Statistik Pasar GSS", "🎯 Evaluasi Model"]
)


# ===========================================================================
# TAB 1 — Klasifikasi prospektus
# ===========================================================================

def render_result(res, source_label: str):
    label, color = CLASS_VIEW.get(res.gss_class.value, (res.gss_class.value, "#000"))
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{color}1a;"
        f"border-left:6px solid {color};'>"
        f"<span style='font-size:1.4rem;font-weight:700;color:{color}'>{label}</span>"
        f"<br><span style='color:#444'>Keyakinan: <b>{res.confidence:.0%}</b> · "
        f"{source_label}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    title_map = {True: "✅ Ya (nama obligasi GSS)", False: "❌ Tidak", None: "❔ Tidak diketahui"}
    c1.metric("Sinyal nama (IDX)", title_map.get(res.title_gss, "—"))
    c2.metric("Sinyal framing badan dok", "✅ Ada" if res.framing_body else "—")
    c3.metric("Ambang dipakai", f"{res.threshold_used:.2f}")

    if res.level0_evidence:
        st.info(f"**Struktur instrumen (Level 0):** {', '.join(res.level0_evidence)}")

    # Sektor eligible + skor
    sect_rows = []
    for key, score in (res.top_env + res.top_soc):
        c = CAT.get(key)
        sect_rows.append({"Sektor": c.name_id if c else key,
                          "Bucket": "Lingkungan" if c and c.bucket.value == "environmental" else "Sosial",
                          "Skor similarity": round(score, 3)})
    if sect_rows:
        st.subheader("Sektor eligible terpenuhi (use-of-proceeds)")
        sdf = pd.DataFrame(sect_rows).sort_values("Skor similarity", ascending=False)
        st.dataframe(sdf, hide_index=True, use_container_width=True)
        st.bar_chart(sdf.set_index("Sektor")["Skor similarity"])
    elif res.gss_class != GSSClass.NON_GSS and not res.level0_evidence:
        st.write("")

    if res.blue:
        st.caption(f"🌊 Sub-tema **Blue** (kelautan): {', '.join(res.blue[:6])}")

    with st.expander("Bukti & penjelasan (explainable by design)"):
        if res.framing_title:
            st.write("**Nama obligasi GSS di listing IDX:**")
            for t in res.framing_title[:8]:
                st.write(f"- {t}")
        if res.framing_body:
            st.write(f"**Frasa framing di badan dokumen:** {', '.join(res.framing_body[:10])}")
        st.write(f"**Anchor bagian Penggunaan Dana:** `{res.anchor}`")
        if res.scores:
            allsc = pd.DataFrame(
                [{"Kategori": CAT[k].name_id if k in CAT else k, "Skor": round(v, 3)}
                 for k, v in res.scores.items()]
            ).sort_values("Skor", ascending=False)
            st.write("**Skor similarity seluruh kategori:**")
            st.dataframe(allsc, hide_index=True, use_container_width=True)


with tab_clf:
    st.markdown("Unggah **prospektus PDF** atau tempel teks. Pilih kode emiten "
                "untuk mengaktifkan *title-lookup* (sinyal terkuat).")

    colL, colR = st.columns([2, 1])
    with colR:
        issuers = ["(tidak diketahui)"] + issuer_list()
        sel = st.selectbox("Kode emiten (opsional)", issuers,
                           help="Jika emiten ada di listing IDX, nama obligasinya "
                                "dipakai sebagai sinyal klasifikasi utama.")
        issuer = None if sel == "(tidak diketahui)" else sel
        if issuer:
            tg, hits = has_gss_title(issuer)
            if tg:
                st.success(f"{issuer}: {len(hits)} instrumen bernama GSS di listing")
            elif tg is False:
                st.warning(f"{issuer}: ada di listing, tak ada instrumen bernama GSS")

    with colL:
        up = st.file_uploader("Prospektus PDF", type=["pdf"])
        txt = st.text_area("…atau tempel teks prospektus", height=140,
                           placeholder="Tempel bagian Penggunaan Dana / Use of Proceeds…")

    if st.button("Klasifikasikan", type="primary"):
        _warm_model()
        text = ""
        src = ""
        if up is not None:
            text = read_pdf_bytes(up.getvalue())
            src = f"sumber: {up.name}"
            if not text.strip():
                st.error("PDF tidak menghasilkan teks (kemungkinan hasil scan/gambar → perlu OCR).")
        elif txt.strip():
            text = txt
            src = "sumber: teks ditempel"
        else:
            st.warning("Unggah PDF atau tempel teks dulu.")

        if text.strip():
            with st.spinner("Menganalisis…"):
                res = classify_ml(text, issuer=issuer)
            if res is None:
                st.error("Teks kosong.")
            else:
                render_result(res, src)
                seg, _ = find_use_of_proceeds(text)
                with st.expander("Cuplikan bagian Penggunaan Dana yang diekstrak"):
                    st.text(seg[:2500])


# ===========================================================================
# TAB 2 — Statistik pasar
# ===========================================================================

with tab_market:
    df = load_universe()
    if df.empty:
        st.error("Data listing IDX tidak ditemukan di data/.")
    else:
        total = len(df)
        gss = df[df["is_gss"]]
        n_gss = len(gss)
        val_gss = gss["Outstanding"].sum()
        val_all = df["Outstanding"].sum()

        st.subheader("Semesta EBUS korporasi (listing IDX)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total instrumen", f"{total:,}")
        m2.metric("Terdeteksi GSS (by name)", f"{n_gss:,}", f"{n_gss/total:.1%}")
        m3.metric("Nilai GSS (outstanding)", f"Rp {val_gss/1e12:,.1f} T")
        m4.metric("Porsi nilai GSS", f"{(val_gss/val_all if val_all else 0):.1%}")

        st.caption(
            "Deteksi berbasis **nama instrumen** (label POJK 18/2023). "
            "Memisahkan jebakan *Penawaran Umum Berkelanjutan* (PUB) dari "
            "*Keberlanjutan/Berwawasan* yang benar-benar GSS."
        )

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Komposisi GSS per tipe (jumlah seri)**")
            by_type = (gss["gss_type"].map(TYPE_LABEL).value_counts().rename_axis("Tipe")
                       .reset_index(name="Jumlah"))
            st.bar_chart(by_type.set_index("Tipe")["Jumlah"])
        with cc2:
            st.markdown("**Nilai outstanding per tipe (Rp triliun)**")
            by_val = (gss.assign(T=gss["Outstanding"] / 1e12)
                      .groupby(gss["gss_type"].map(TYPE_LABEL))["T"].sum()
                      .rename_axis("Tipe").reset_index(name="Rp triliun"))
            st.bar_chart(by_val.set_index("Tipe")["Rp triliun"])

        st.subheader("Daftar instrumen GSS terdeteksi")
        q = st.text_input("Cari (nama / kode emiten)", "")
        show = gss.copy()
        show["Tipe"] = show["gss_type"].map(TYPE_LABEL)
        show["Outstanding (Rp M)"] = (show["Outstanding"] / 1e9).round(1)
        cols = ["IssuerCode", "BondName", "Tipe", "Rating", "Outstanding (Rp M)"]
        cols = [c for c in cols if c in show.columns]
        if q:
            mask = (show["BondName"].str.contains(q, case=False, na=False) |
                    show["IssuerCode"].str.contains(q, case=False, na=False))
            show = show[mask]
        st.dataframe(show[cols].sort_values("Outstanding (Rp M)", ascending=False),
                     hide_index=True, use_container_width=True, height=400)
        st.caption(f"{len(show):,} instrumen ditampilkan.")


# ===========================================================================
# TAB 3 — Evaluasi
# ===========================================================================

with tab_eval:
    cmp = load_comparison()
    if cmp.empty:
        st.info("Belum ada data evaluasi. Jalankan `python evaluation/compare_engines.py`.")
    else:
        st.subheader("Hasil evaluasi terhadap gold set")
        st.caption("27 dokumen GSS terverifikasi + 66 dokumen konvensional (sampel).")

        def metrics(pred_col):
            tp = ((cmp["gold"] == "GSS") & (cmp[pred_col] == "GSS")).sum()
            fp = ((cmp["gold"] != "GSS") & (cmp[pred_col] == "GSS")).sum()
            fn = ((cmp["gold"] == "GSS") & (cmp[pred_col] != "GSS")).sum()
            tn = ((cmp["gold"] != "GSS") & (cmp[pred_col] != "GSS")).sum()
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            return p, r, f1, tp, fp, fn, tn

        pr, rr, fr, *_ = metrics("pred_rule")
        pm, rm, fm, tpm, fpm, fnm, tnm = metrics("pred_ml")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Rule-based")
            st.metric("Precision", f"{pr:.2f}")
            st.metric("Recall", f"{rr:.2f}")
            st.metric("F1", f"{fr:.2f}")
        with c2:
            st.markdown("#### ML + Title Gate")
            st.metric("Precision", f"{pm:.2f}", f"{pm-pr:+.2f}")
            st.metric("Recall", f"{rm:.2f}", f"{rm-rr:+.2f}")
            st.metric("F1", f"{fm:.2f}", f"{fm-fr:+.2f}")

        st.write(f"**ML confusion:** TP={tpm} · FP={fpm} · FN={fnm} · TN={tnm}")

        st.subheader("Detail per dokumen")
        show_cols = [c for c in ["issuer", "gold", "pred_rule", "pred_ml",
                                 "ml_class", "ml_title_gss", "pdf"] if c in cmp.columns]
        only_err = st.checkbox("Tampilkan hanya yang salah (ML)", False)
        view = cmp[cmp["gold"] != cmp["pred_ml"]] if only_err else cmp
        st.dataframe(view[show_cols], hide_index=True, use_container_width=True, height=460)
