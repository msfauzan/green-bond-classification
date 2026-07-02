"""
Dashboard (Streamlit) — Klasifikasi EBUS GSS, Bank Indonesia (DSta/DSMF).

Tiga halaman (navigasi sidebar):
  1. Statistik Pasar GSS      — KPI + grafik semesta 1.437 EBUS IDX
  2. Klasifikasi Prospektus   — unggah PDF / tempel teks → kelas GSS + bukti
  3. Evaluasi Model           — metrik & hasil per-dokumen (gold set)

Jalankan dari root repo:
    streamlit run code/webapp/app.py

Semua lokal & bebas (sentence-transformers + PyMuPDF). Tanpa API berbayar.
"""
from __future__ import annotations

import os
import sys
import datetime

import altair as alt
import pandas as pd
import streamlit as st

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../code
if CODE not in sys.path:
    sys.path.insert(0, CODE)
ROOT = os.path.dirname(CODE)                                        # repo root
DATA = os.path.join(ROOT, "data")

from classifier.engine import read_pdf_bytes, find_use_of_proceeds
from classifier.ml_engine import classify_ml
from classifier.taxonomy import GSSClass, category_index

from classifier.title_lookup import (
    all_instruments, gss_type_from_title, has_gss_title, _load_idx,
)

CAT = category_index()

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
    "sustainability_linked": "Sustainability-Linked",
}
TYPE_COLOR = {
    "Green": "#16a34a", "Social": "#ea580c",
    "Sustainability": "#2563eb", "Sustainability-Linked": "#9333ea",
}

st.set_page_config(
    page_title="Dashboard EBUS GSS — Bank Indonesia",
    page_icon="🌱", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetric"] {
      background: var(--secondary-background-color);
      border: 1px solid rgba(128,128,128,.25);
      border-radius: 10px; padding: 12px 16px;
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
  h1, h2, h3 { letter-spacing: -.01em; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cache: model & data
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Memuat model semantik (sekali saja)…")
def _warm_model():
    from classifier.ml_engine import _get_model, _category_embeddings
    _get_model()
    _category_embeddings()
    return True


GOLD_DB = os.path.join(DATA, "gold_bonds_db.csv")


@st.cache_data
def load_universe() -> pd.DataFrame:
    rows = all_instruments()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["gss_type"] = df["BondName"].map(gss_type_from_title)
    df["is_gss"] = df["gss_type"].notna()
    df["Outstanding"] = pd.to_numeric(df.get("Outstanding"), errors="coerce").fillna(0)
    df["MatureDate"] = pd.to_datetime(df.get("MatureDate"), errors="coerce", utc=True).dt.tz_localize(None)
    today = pd.Timestamp(datetime.date.today())
    df["Aktif"] = df["MatureDate"].apply(lambda d: pd.isna(d) or d >= today)
    df["Source"] = "IDX Listing"
    return df


@st.cache_data
def load_gold_db() -> pd.DataFrame:
    """Muat gold_bonds_db.csv — instrumen dari gold set yang TIDAK ada di IDX listing."""
    if not os.path.exists(GOLD_DB):
        return pd.DataFrame()
    gdf = pd.read_csv(GOLD_DB, encoding="utf-8-sig")
    gdf = gdf[gdf["Source"] == "Gold Dataset"].copy()
    gdf["Outstanding"] = pd.to_numeric(gdf.get("Outstanding"), errors="coerce").fillna(0)
    gdf["MatureDate"] = pd.to_datetime(gdf.get("MatureDate"), errors="coerce")
    gdf["Aktif"] = False  # semua Gold-only → tidak ada di IDX saat scraping
    gdf["gss_type"] = gdf["GSSType"].where(gdf["GSSType"].notna() & (gdf["GSSType"] != ""),
                                            gdf["BondName"].map(gss_type_from_title))
    gdf["is_gss"] = True
    return gdf


@st.cache_data
def load_comparison() -> pd.DataFrame:
    p = os.path.join(DATA, "comparison_results.csv")
    return pd.read_csv(p, encoding="utf-8-sig") if os.path.exists(p) else pd.DataFrame()


@st.cache_data
def issuer_list() -> list[str]:
    return sorted(_load_idx().keys())


def type_chart_data(gss: pd.DataFrame) -> pd.DataFrame:
    g = gss.assign(Tipe=gss["gss_type"].map(TYPE_LABEL))
    out = (g.groupby("Tipe")
             .agg(Jumlah=("Tipe", "size"), Nilai=("Outstanding", "sum"))
             .reset_index())
    out["Rp triliun"] = (out["Nilai"] / 1e12).round(2)
    return out


# ---------------------------------------------------------------------------
# Sidebar — navigasi & identitas
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌱 EBUS GSS")
    st.caption("**Bank Indonesia** · DSta / DSMF")
    page = st.radio(
        "Navigasi",
        ["📊 Statistik Pasar", "🔎 Klasifikasi Prospektus", "🎯 Evaluasi Model"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Klasifikasi Efek Bersifat Utang & Sukuk ke kategori "
        "**Green / Social / Sustainability** (POJK 18/2023, ICMA, DJPPR).\n\n"
        "Pipeline *taxonomy-grounded*: rule-based + ML semantic, "
        "sepenuhnya lokal & explainable."
    )


# ===========================================================================
# HALAMAN — Statistik Pasar (dashboard utama)
# ===========================================================================

def page_market():
    st.title("📊 Statistik Pasar EBUS GSS")

    df = load_universe()
    gdf = load_gold_db()
    if df.empty:
        st.error("Data listing IDX tidak ditemukan di data/.")
        return

    gss_idx = df[df["is_gss"]]
    gss_aktif = gss_idx[gss_idx["Aktif"]]
    val_gss = gss_aktif["Outstanding"].sum()
    val_all = df[df["Aktif"]]["Outstanding"].sum()
    n_gold_only = len(gdf)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Instrumen aktif (IDX)", f"{len(df[df['Aktif']]):,}")
    m2.metric("GSS aktif (IDX)", f"{len(gss_aktif):,}")
    m3.metric("GSS jatuh tempo (Gold)", f"{n_gold_only:,}",
              help="Instrumen di gold dataset kita yang sudah tidak ada di IDX listing")
    m4.metric("Nilai GSS outstanding", f"Rp {val_gss/1e12:,.1f} T")
    m5.metric("Porsi nilai GSS", f"{(val_gss/val_all if val_all else 0):.1%}",
              help="Terhadap seluruh EBUS korporasi aktif")

    st.caption(
        "Deteksi berbasis **nama instrumen** (label POJK 18/2023). "
        "Memisahkan jebakan *Penawaran Umum Berkelanjutan* (PUB) dari "
        "*Keberlanjutan/Berwawasan* yang benar-benar GSS. "
        "**Gold Dataset** = dari gold set prospektus, sudah tidak di IDX listing."
    )
    st.write("")

    tdata = type_chart_data(gss_aktif)
    c1, c2, c3 = st.columns([1, 1, 1.2])

    color_scale = alt.Scale(domain=list(TYPE_COLOR.keys()),
                            range=list(TYPE_COLOR.values()))
    CHART_H = 300
    with c1:
        st.markdown("**Komposisi GSS aktif (jumlah seri)**")
        donut = (alt.Chart(tdata).mark_arc(innerRadius=60)
                 .encode(theta="Jumlah:Q",
                         color=alt.Color("Tipe:N", scale=color_scale,
                                         legend=alt.Legend(orient="bottom", columns=2,
                                                           labelLimit=200)),
                         tooltip=["Tipe", "Jumlah"])
                 .properties(height=CHART_H))
        st.altair_chart(donut, use_container_width=True)
    with c2:
        st.markdown("**Nilai outstanding per tipe (Rp T)**")
        bar = (alt.Chart(tdata).mark_bar(cornerRadius=4)
               .encode(x=alt.X("Rp triliun:Q"),
                       y=alt.Y("Tipe:N", sort="-x", title=None,
                               axis=alt.Axis(labelLimit=180)),
                       color=alt.Color("Tipe:N", scale=color_scale, legend=None),
                       tooltip=["Tipe", "Rp triliun"])
               .properties(height=CHART_H))
        st.altair_chart(bar, use_container_width=True)
    with c3:
        st.markdown("**Emiten GSS terbesar (outstanding aktif, Rp T)**")
        top_iss = (gss_aktif.groupby("IssuerCode")["Outstanding"].sum()
                   .sort_values(ascending=False).head(10) / 1e12).round(2)
        tdf = top_iss.rename_axis("Emiten").reset_index(name="Rp triliun")
        ibar = (alt.Chart(tdf).mark_bar(cornerRadius=4, color="#0e7490")
                .encode(x="Rp triliun:Q",
                        y=alt.Y("Emiten:N", sort="-x", title=None),
                        tooltip=["Emiten", "Rp triliun"])
                .properties(height=CHART_H))
        st.altair_chart(ibar, use_container_width=True)

    st.markdown("**Profil jatuh tempo GSS aktif (Rp T per tahun)**")
    mat = gss_aktif.dropna(subset=["MatureDate"]).copy()
    mat["Tahun"] = mat["MatureDate"].dt.year.astype(str)
    mat["Tipe"] = mat["gss_type"].map(TYPE_LABEL)
    mdata = (mat.groupby(["Tahun", "Tipe"])["Outstanding"].sum() / 1e12).round(2)
    mdata = mdata.reset_index(name="Rp triliun")
    mchart = (alt.Chart(mdata).mark_bar(cornerRadius=3)
              .encode(x=alt.X("Tahun:O", title=None),
                      y="Rp triliun:Q",
                      color=alt.Color("Tipe:N", scale=color_scale,
                                      legend=alt.Legend(orient="top")),
                      tooltip=["Tahun", "Tipe", "Rp triliun"])
              .properties(height=280))
    st.altair_chart(mchart, use_container_width=True)

    # ------------------------------------------------------------------ tabel
    st.subheader("Daftar instrumen GSS")
    qcol, tcol, fcol = st.columns([2, 1, 1])
    q = qcol.text_input("Cari (nama / kode emiten)", "")
    type_filter = tcol.multiselect("Tipe", list(TYPE_COLOR.keys()), default=[])
    status_filter = fcol.radio("Status", ["Semua", "Aktif saja", "Sudah jatuh tempo"],
                               horizontal=True)

    show_idx = gss_idx.copy()
    show_idx["Tipe"] = show_idx["gss_type"].map(TYPE_LABEL)
    show_idx["Outstanding (Rp M)"] = (show_idx["Outstanding"] / 1e9).round(1)
    show_idx["Jatuh Tempo"] = show_idx["MatureDate"].dt.strftime("%d %b %Y").fillna("—")
    show_idx["Status"] = show_idx["Aktif"].map({True: "🟢 Aktif", False: "🔴 Jatuh tempo"})

    common_cols = ["IssuerCode", "BondName", "Tipe", "Rating",
                   "Outstanding (Rp M)", "Jatuh Tempo", "Status", "Aktif", "Source"]
    if not gdf.empty:
        show_gold = gdf.copy()
        show_gold["Tipe"] = show_gold["gss_type"].map(TYPE_LABEL).fillna("—")
        show_gold["Outstanding (Rp M)"] = 0.0
        show_gold["Jatuh Tempo"] = "—"
        show_gold["Status"] = "🔴 Jatuh tempo (Gold)"
        show = pd.concat([show_idx[common_cols], show_gold[common_cols]], ignore_index=True)
    else:
        show = show_idx[common_cols]

    if status_filter == "Aktif saja":
        show = show[show["Aktif"]]
    elif status_filter == "Sudah jatuh tempo":
        show = show[~show["Aktif"]]
    if type_filter:
        show = show[show["Tipe"].isin(type_filter)]
    if q:
        mask = (show["BondName"].str.contains(q, case=False, na=False) |
                show["IssuerCode"].str.contains(q, case=False, na=False))
        show = show[mask]

    show = show.sort_values(["Aktif", "Outstanding (Rp M)"], ascending=[False, False])
    cols = [c for c in common_cols if c != "Aktif"]
    st.dataframe(show[cols], hide_index=True, use_container_width=True, height=440)
    n_aktif_show = int(show["Aktif"].sum())
    st.caption(f"{len(show):,} instrumen ditampilkan · {n_aktif_show:,} aktif (IDX), "
               f"{len(show) - n_aktif_show:,} sudah jatuh tempo.")
    st.download_button("⬇️ Unduh CSV", show[cols].to_csv(index=False).encode("utf-8-sig"),
                       "gss_instruments.csv", "text/csv")


# ===========================================================================
# HALAMAN — Klasifikasi prospektus
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

    c1, c2, c3, c4 = st.columns(4)
    title_map = {True: "✅ Ya (nama obligasi GSS)", False: "❌ Tidak", None: "❔ Tidak diketahui"}
    c1.metric("Sinyal nama (IDX)", title_map.get(res.title_gss, "—"))
    c2.metric("Sinyal framing badan dok", "✅ Ada" if res.framing_body else "—")
    c3.metric("Ambang dipakai", f"{res.threshold_used:.2f}")
    series_label = ", ".join(f"Seri {s}" for s in res.series) if res.series else "—"
    c4.metric("Seri di sampul", series_label)

    if res.level0_evidence:
        st.info(f"**Struktur instrumen (Level 0):** {', '.join(res.level0_evidence)}")

    sect_rows = []
    for key, score in (res.top_env + res.top_soc):
        c = CAT.get(key)
        sect_rows.append({"Sektor": c.name_id if c else key,
                          "Bucket": "Lingkungan" if c and c.bucket.value == "environmental" else "Sosial",
                          "Skor similarity": round(score, 3)})
    if sect_rows:
        st.subheader("Sektor eligible terpenuhi (use-of-proceeds)")
        sdf = pd.DataFrame(sect_rows).sort_values("Skor similarity", ascending=False)
        tcol, ccol = st.columns([1, 1])
        tcol.dataframe(sdf, hide_index=True, use_container_width=True)
        schart = (alt.Chart(sdf).mark_bar(cornerRadius=4)
                  .encode(x="Skor similarity:Q",
                          y=alt.Y("Sektor:N", sort="-x", title=None),
                          color=alt.Color("Bucket:N",
                                          scale=alt.Scale(domain=["Lingkungan", "Sosial"],
                                                          range=["#16a34a", "#ea580c"])),
                          tooltip=["Sektor", "Bucket", "Skor similarity"]))
        ccol.altair_chart(schart, use_container_width=True)

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


def page_classify():
    st.title("🔎 Klasifikasi Prospektus")
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
# HALAMAN — Evaluasi model
# ===========================================================================

def page_eval():
    st.title("🎯 Evaluasi Model")
    cmp = load_comparison()
    if cmp.empty:
        st.info("Belum ada data evaluasi. Jalankan `python code/evaluation/compare_engines.py`.")
        return

    n_pos = (cmp["gold"] == "GSS").sum()
    n_neg = (cmp["gold"] != "GSS").sum()
    st.caption(f"Gold set: **{n_pos} dokumen GSS terverifikasi** + "
               f"**{n_neg} dokumen konvensional** (sampel).")

    def metrics(pred_col):
        tp = ((cmp["gold"] == "GSS") & (cmp[pred_col] == "GSS")).sum()
        fp = ((cmp["gold"] != "GSS") & (cmp[pred_col] == "GSS")).sum()
        fn = ((cmp["gold"] == "GSS") & (cmp[pred_col] != "GSS")).sum()
        tn = ((cmp["gold"] != "GSS") & (cmp[pred_col] != "GSS")).sum()
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        return p, r, f1, tp, fp, fn, tn

    pr, rr, fr, tpr, fpr, fnr, tnr = metrics("pred_rule")
    pm, rm, fm, tpm, fpm, fnm, tnm = metrics("pred_ml")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚙️ Rule-based (baseline)")
        a, b, c = st.columns(3)
        a.metric("Precision", f"{pr:.2f}")
        b.metric("Recall", f"{rr:.2f}")
        c.metric("F1", f"{fr:.2f}")
        st.caption(f"Confusion: TP={tpr} · FP={fpr} · FN={fnr} · TN={tnr}")
    with c2:
        st.markdown("#### 🧠 ML Semantic + Title Gate")
        a, b, c = st.columns(3)
        a.metric("Precision", f"{pm:.2f}", f"{pm-pr:+.2f}")
        b.metric("Recall", f"{rm:.2f}", f"{rm-rr:+.2f}")
        c.metric("F1", f"{fm:.2f}", f"{fm-fr:+.2f}")
        st.caption(f"Confusion: TP={tpm} · FP={fpm} · FN={fnm} · TN={tnm}")

    st.write("")
    mcol, dcol = st.columns([1, 1])
    with mcol:
        st.markdown("**Perbandingan metrik**")
        mdf = pd.DataFrame({
            "Metrik": ["Precision", "Recall", "F1"] * 2,
            "Engine": ["Rule-based"] * 3 + ["ML + Title Gate"] * 3,
            "Nilai": [pr, rr, fr, pm, rm, fm],
        })
        mchart = (alt.Chart(mdf).mark_bar(cornerRadius=4)
                  .encode(x=alt.X("Metrik:N", title=None, sort=["Precision", "Recall", "F1"],
                                  axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                          xOffset=alt.XOffset("Engine:N",
                                              sort=["Rule-based", "ML + Title Gate"]),
                          y=alt.Y("Nilai:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                          color=alt.Color("Engine:N",
                                          scale=alt.Scale(domain=["Rule-based", "ML + Title Gate"],
                                                          range=["#94a3b8", "#2563eb"]),
                                          legend=alt.Legend(orient="top", title=None,
                                                            labelLimit=220)),
                          tooltip=["Engine", "Metrik", alt.Tooltip("Nilai:Q", format=".2f")])
                  .properties(height=280))
        st.altair_chart(mchart, use_container_width=True)
    with dcol:
        if "ml_class" in cmp.columns:
            st.markdown("**Sub-kelas ML pada prediksi GSS benar (TP)**")
            sub = (cmp[(cmp["gold"] == "GSS") & (cmp["pred_ml"] == "GSS")]["ml_class"]
                   .value_counts().rename_axis("Kelas").reset_index(name="Jumlah"))
            sub["Kelas"] = sub["Kelas"].map(
                {k: v[0].split("—")[0].strip() for k, v in CLASS_VIEW.items()}
            ).fillna(sub["Kelas"])
            schart = (alt.Chart(sub).mark_bar(cornerRadius=4, color="#16a34a")
                      .encode(x="Jumlah:Q",
                              y=alt.Y("Kelas:N", sort="-x", title=None,
                                      axis=alt.Axis(labelLimit=220)),
                              tooltip=["Kelas", "Jumlah"])
                      .properties(height=280))
            st.altair_chart(schart, use_container_width=True)

    st.subheader("Detail per dokumen")
    show_cols = [c for c in ["issuer", "gold", "pred_rule", "pred_ml",
                             "ml_class", "ml_title_gss", "ml_conf", "pdf"] if c in cmp.columns]
    only_err = st.checkbox("Tampilkan hanya yang salah (ML)", False)
    view = cmp[cmp["gold"] != cmp["pred_ml"]] if only_err else cmp
    st.dataframe(view[show_cols], hide_index=True, use_container_width=True, height=460)


# ===========================================================================
# Router
# ===========================================================================

if page.startswith("📊"):
    page_market()
elif page.startswith("🔎"):
    page_classify()
else:
    page_eval()
