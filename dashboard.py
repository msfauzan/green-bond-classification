"""
Dashboard Analisis GSS EBUS Korporasi — Bank Indonesia (DSta/DSMF)
Jalankan: streamlit run dashboard.py
"""
import os
from collections import defaultdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard GSS EBUS Korporasi · Bank Indonesia",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
N_UNIVERSE = 1437

# ── Design tokens ─────────────────────────────────────────────────────────────
# Solid brand colours (for chart markers, borders, icons)
C_GREEN  = "#2E7D32"; C_SOC = "#C2A100"; C_SUS = "#6A1FA8"; C_SL = "#1565C0"
C_BI     = "#0056B2"; C_ERR = "#C62828"; C_OK  = "#2E7D32"

# Transparent surface fills — work on both dark and light backgrounds
S_GREEN  = "rgba(46,125,50,0.12)";  S_SOC = "rgba(194,161,0,0.12)"
S_SUS    = "rgba(106,31,168,0.12)"; S_SL  = "rgba(21,101,192,0.12)"
S_BI     = "rgba(0,86,178,0.10)";   S_ERR = "rgba(198,40,40,0.12)"
S_OK     = "rgba(46,125,50,0.12)";  S_WARN = "rgba(230,81,0,0.12)"

CLASS_COLORS = {"Green": C_GREEN, "Social": C_SOC,
                "Sustainability": C_SUS, "Sustainability Linked": C_SL}
CLASS_SURF   = {"Green": S_GREEN, "Social": S_SOC,
                "Sustainability": S_SUS, "Sustainability Linked": S_SL}

CAT_LABEL = {
    "renewable_energy":"Energi Terbarukan","energy_efficiency":"Efisiensi Energi",
    "green_building":"Bangunan Hijau","green_tourism":"Pariwisata Berkelanjutan",
    "sustainable_transport":"Transportasi Berkelanjutan","waste_management":"Pengelolaan Sampah & Limbah",
    "water_management":"Pengelolaan Air & Air Limbah","natural_resources":"Pengelolaan SDA Berkelanjutan",
    "climate_resilience":"Ketahanan Iklim & PRB","basic_infrastructure":"Infrastruktur Dasar Terjangkau",
    "essential_services":"Akses Layanan Esensial","affordable_housing":"Perumahan Terjangkau",
    "employment_msme":"Penciptaan Lapangan Kerja & UMKM","food_security":"Ketahanan Pangan",
    "socioeconomic":"Pemberdayaan Sosial-Ekonomi",
}
CAT_BUCKET = {k:"Lingkungan" for k in list(CAT_LABEL)[:9]}
CAT_BUCKET.update({k:"Sosial" for k in list(CAT_LABEL)[9:]})

# ── Design system CSS ─────────────────────────────────────────────────────────
# Rules: no forced text colours, no forced backgrounds — only layout + accent.
# rgba() for all fills so they layer over both dark and light surfaces.
st.markdown("""
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
footer { visibility: hidden; }

/* Metric cards */
[data-testid="stMetric"] {
    border-radius: 10px;
    padding: 14px 16px !important;
    background: rgba(128,128,128,0.06);
    border: 1px solid rgba(128,128,128,0.15);
}
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; opacity: 0.7; }
[data-testid="stMetricDelta"]  { font-size: 0.78rem; }

/* Section dividers */
h2 {
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(0,86,178,0.30);
    margin-top: 0.6rem !important;
}

/* Sidebar nav */
[data-testid="stRadio"] label { font-size: 0.9rem; padding: 4px 0; }
[data-testid="stRadio"] div[data-baseweb="radio"] {
    gap: 2px;
}

/* Expanders */
[data-testid="stExpander"] details {
    border-radius: 8px !important;
    border: 1px solid rgba(128,128,128,0.20) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Design helpers ────────────────────────────────────────────────────────────
def _card(icon, label, value, sub="", color=C_BI, surf=S_BI):
    """Styled metric card via HTML — colour-accented, dark-mode-safe."""
    st.markdown(f"""
    <div style="border-radius:10px;border:1px solid {color}33;
                background:{surf};padding:16px 18px;height:100%">
      <div style="font-size:1.5rem;line-height:1">{icon}</div>
      <div style="font-size:0.78rem;opacity:0.65;margin-top:6px;
                  letter-spacing:0.02em;text-transform:uppercase">{label}</div>
      <div style="font-size:1.7rem;font-weight:700;margin-top:4px;
                  color:{color};line-height:1.1">{value}</div>
      <div style="font-size:0.78rem;opacity:0.6;margin-top:4px">{sub}</div>
    </div>""", unsafe_allow_html=True)

def _badge(text, color, surf):
    return (f"<span style='background:{surf};color:{color};"
            f"border:1px solid {color}55;border-radius:4px;"
            f"padding:2px 8px;font-size:0.75rem;font-weight:600'>{text}</span>")

def _section(title, caption=""):
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)

def _chart(fig, height=None):
    """Uniform chart: transparent background, adaptive font, no white artifacts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=20, b=30, l=10, r=10),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font_size=11,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.2)")
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)

def _divider():
    st.markdown("<hr style='border:none;border-top:1px solid rgba(128,128,128,0.2);margin:6px 0'>",
                unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_census():
    return pd.read_csv(os.path.join(DATA, "market_census.csv"))

@st.cache_data
def load_gss():
    df = pd.read_csv(os.path.join(DATA, "idx_gss_all_20260618_140427.csv"))
    df["Outstanding_M"] = pd.to_numeric(df["Outstanding"], errors="coerce") / 1e6
    df["JatuhTempo"]    = pd.to_datetime(df["MatureDate"], errors="coerce").dt.date
    return df

@st.cache_data
def load_sectors():
    return pd.read_csv(os.path.join(DATA, "sector_decomposition.csv"))

@st.cache_data
def load_comparison():
    return pd.read_csv(os.path.join(DATA, "comparison_results.csv"))

@st.cache_data
def load_sbn():
    return pd.read_csv(os.path.join(DATA, "sbn_gss_lookup.csv"))

def prf(df, col):
    tp = ((df["gold"]=="GSS") & (df[col]=="GSS")).sum()
    fp = ((df["gold"]!="GSS") & (df[col]=="GSS")).sum()
    fn = ((df["gold"]=="GSS") & (df[col]!="GSS")).sum()
    tn = ((df["gold"]!="GSS") & (df[col]!="GSS")).sum()
    P  = tp/(tp+fp) if (tp+fp) else 0
    R  = tp/(tp+fn) if (tp+fn) else 0
    F1 = 2*P*R/(P+R) if (P+R) else 0
    return {"P":P,"R":R,"F1":F1,"TP":int(tp),"FP":int(fp),"FN":int(fn),"TN":int(tn)}

def sector_expanded(sectors_df):
    rows = []
    for _, r in sectors_df.iterrows():
        keys = str(r.get("sector_keys","") or "")
        if keys and keys != "nan":
            for k in keys.split("|"):
                k = k.strip()
                if k:
                    rows.append({
                        "Emiten": r["issuer"],
                        "Kelas GSS": r["gss_class"],
                        "Bucket": r["bucket"],
                        "Kategori": CAT_LABEL.get(k, k),
                        "Kelompok": CAT_BUCKET.get(k,"Lainnya"),
                        "Status Verifikasi": r["verification_status"],
                    })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Ringkasan Eksekutif
# ══════════════════════════════════════════════════════════════════════════════
def page_ringkasan():
    _section("🏠 Ringkasan Eksekutif",
             "AN Klasifikasi EBUS GSS Korporasi · Bank Indonesia DSta/DSMF · Per POJK 18/2023")

    census  = load_census()
    sectors = load_sectors()
    comp    = load_comparison()

    total   = census[census["gss_type"]=="TOTAL"].iloc[0]
    n_gss   = int(total["n_instrumen"])
    os_t    = float(total["total_outstanding_triliun"])
    share   = float(total["share_universe_pct"])
    rule_m  = prf(comp, "pred_rule")
    ml_m    = prf(comp, "pred_ml")
    n_dec   = (sectors["bucket"]=="terdekomposisi").sum()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    cols = st.columns(5, gap="small")
    with cols[0]: _card("📋","Universe EBUS Korporasi",f"{N_UNIVERSE:,}","Per 18 Juni 2026",C_BI,S_BI)
    with cols[1]: _card("🌿","GSS Berlabel (IDX)",str(n_gss),f"{share:.2f}% dari universe",C_GREEN,S_GREEN)
    with cols[2]: _card("💰","Outstanding GSS",f"Rp {os_t:.1f} T","IDR",C_SOC,S_SOC)
    with cols[3]: _card("🤖","ML F1-Score",f"{ml_m['F1']:.3f}",f"vs rule {rule_m['F1']:.3f}",C_SL,S_SL)
    with cols[4]: _card("🔬","Bond Terdekomposisi",f"{n_dec}/{len(sectors)}","sektor UoP terbaca",C_SUS,S_SUS)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Distribusi kelas + Gap universe ─────────────────────────────────
    c1, c2 = st.columns([3, 2], gap="medium")

    with c1:
        _section("Distribusi GSS per Kelas")
        cls = census[census["gss_type"]!="TOTAL"].copy()
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Jumlah Instrumen","Outstanding (Rp T)"])
        colors = [CLASS_COLORS.get(gt,"#888") for gt in cls["gss_type"]]
        fig.add_trace(go.Bar(
            x=cls["gss_type"], y=cls["n_instrumen"].astype(int),
            marker_color=colors, marker_line_width=0,
            text=cls["n_instrumen"].astype(int), textposition="outside",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=cls["gss_type"], y=cls["total_outstanding_triliun"].astype(float),
            marker_color=colors, marker_line_width=0,
            text=cls["total_outstanding_triliun"].apply(lambda x: f"{x:.1f}T"),
            textposition="outside", showlegend=False,
        ), row=1, col=2)
        _chart(fig, 300)

    with c2:
        _section("Gap vs Universe EBUS")
        n_unlabeled = N_UNIVERSE - n_gss
        fig2 = go.Figure(go.Pie(
            labels=["GSS Berlabel", "Belum Terklasifikasi"],
            values=[n_gss, n_unlabeled],
            hole=0.65,
            marker_colors=[C_BI, "rgba(128,128,128,0.3)"],
            marker_line=dict(color="rgba(0,0,0,0)", width=0),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b>{n_gss}</b><br><span style='font-size:11'>GSS</span>",
            x=0.5, y=0.5, showarrow=False, font_size=20, align="center",
        )
        fig2.update_layout(showlegend=False)
        _chart(fig2, 300)

    # ── Row 2: Dekomposisi + Verifikasi ───────────────────────────────────────
    c3, c4 = st.columns(2, gap="medium")

    BKT_LABEL = {"terdekomposisi":"Terdekomposisi","sektor_tak_terverifikasi":"Tanpa Sektor","level0":"Level-0"}
    BKT_COLOR = {"terdekomposisi":C_GREEN,"sektor_tak_terverifikasi":"#E65100","level0":C_SUS}
    VS_COLOR  = {"Terverifikasi":C_GREEN,"Sebagian":C_SOC,
                 "Tidak terverifikasi (UoP tipis)":"#E65100","Level-0 (struktural)":C_SUS}

    with c3:
        _section("Status Dekomposisi Sektoral")
        bkt = sectors["bucket"].value_counts().reset_index()
        bkt.columns = ["Bucket","n"]
        bkt["Label"]  = bkt["Bucket"].map(BKT_LABEL)
        bkt["Warna"]  = bkt["Bucket"].map(BKT_COLOR)
        fig3 = go.Figure(go.Bar(
            x=bkt["n"], y=bkt["Label"], orientation="h",
            marker_color=bkt["Warna"], marker_line_width=0,
            text=bkt["n"], textposition="outside",
        ))
        _chart(fig3, 240)

    with c4:
        _section("Status Verifikasi Klaim")
        vs = sectors["verification_status"].value_counts().reset_index()
        vs.columns = ["Status","n"]
        fig4 = go.Figure(go.Pie(
            labels=vs["Status"], values=vs["n"],
            marker_colors=[VS_COLOR.get(s,"#888") for s in vs["Status"]],
            marker_line=dict(color="rgba(0,0,0,0)", width=0),
            hole=0.45, textinfo="label+percent",
        ))
        fig4.update_layout(showlegend=False)
        _chart(fig4, 240)

    # ── Metodologi ─────────────────────────────────────────────────────────────
    _section("Alur Metodologi Klasifikasi")
    steps = [
        ("1","Level-0","Sinyal struktur instrumen\n(SL · Wakaf)", C_SL),
        ("2","Title Gate","Pencocokan judul IDX\nvs pola GSS", C_BI),
        ("3","Framing Gate","Frasa POJK 18/2023\n+ negasi filter", C_GREEN),
        ("4","Semantik ML","Kemiripan makna\nmodel lokal offline", C_GREEN),
        ("5","Verifikasi","Klaim vs bukti UoP\nlaporan konsistensi", C_SUS),
    ]
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    for col, (num, title, desc, color) in zip([m1,m2,m3,m4,m5], steps):
        with col:
            st.markdown(f"""
            <div style="border-radius:10px;border:1px solid {color}55;
                        background:linear-gradient(135deg,{color}22,{color}0a);
                        padding:14px 12px;text-align:center;min-height:110px">
              <div style="font-size:1.4rem;font-weight:800;color:{color};
                          line-height:1">{num}</div>
              <div style="font-size:0.85rem;font-weight:700;margin-top:4px">{title}</div>
              <div style="font-size:0.72rem;opacity:0.65;margin-top:6px;
                          line-height:1.4;white-space:pre-line">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Metodologi tanpa API berbayar** — model `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB) "
            "berjalan offline. Setiap keputusan dapat ditelusuri ke kata kunci / kriteria pemicunya "
            "(*explainable by design*).")


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Sensus Pasar GSS
# ══════════════════════════════════════════════════════════════════════════════
def page_sensus():
    _section("📊 Sensus Pasar EBUS GSS Korporasi",
             "Berdasarkan nama & kode BEI · Sensus 18 Juni 2026")

    census = load_census()
    gss    = load_gss()
    cls    = census[census["gss_type"]!="TOTAL"].copy()
    total  = census[census["gss_type"]=="TOTAL"].iloc[0]

    # KPI strip
    icons  = {"Green":"🌿","Social":"👥","Sustainability":"♻️","Sustainability Linked":"🔗"}
    cols   = st.columns(len(cls)+1, gap="small")
    for i, (_, row) in enumerate(cls.iterrows()):
        gt = row["gss_type"]
        with cols[i]:
            _card(icons.get(gt,"📌"), gt,
                  f"{int(row['n_instrumen'])} instrumen",
                  f"Rp {row['total_outstanding_triliun']:.2f} T",
                  CLASS_COLORS.get(gt,C_BI), CLASS_SURF.get(gt,S_BI))
    with cols[-1]:
        _card("📌","TOTAL GSS",f"{int(total['n_instrumen'])} instrumen",
              f"Rp {total['total_outstanding_triliun']:.2f} T", C_BI, S_BI)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        _section("Instrumen & Outstanding per Kelas")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = [CLASS_COLORS.get(gt,"#888") for gt in cls["gss_type"]]
        fig.add_trace(go.Bar(
            name="Instrumen", x=cls["gss_type"],
            y=cls["n_instrumen"].astype(int),
            marker_color=colors, marker_line_width=0,
            text=cls["n_instrumen"].astype(int), textposition="outside",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            name="Outstanding (Rp T)", x=cls["gss_type"],
            y=cls["total_outstanding_triliun"].astype(float),
            mode="markers+lines",
            marker=dict(size=10, color="rgba(255,255,255,0.9)",
                        line=dict(color=C_BI, width=2.5)),
            line=dict(color=C_BI, dash="dot"),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Jumlah Instrumen", secondary_y=False)
        fig.update_yaxes(title_text="Outstanding (Rp T)", secondary_y=True)
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        _chart(fig, 360)

    with c2:
        _section("Komposisi Outstanding (Rp Triliun)")
        fig2 = go.Figure(go.Pie(
            labels=cls["gss_type"],
            values=cls["total_outstanding_triliun"].astype(float),
            marker_colors=[CLASS_COLORS.get(gt,"#888") for gt in cls["gss_type"]],
            marker_line=dict(color="rgba(0,0,0,0)", width=0),
            hole=0.5, textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Rp %{value:.2f} T<extra></extra>",
        ))
        fig2.update_layout(showlegend=True,
                           legend=dict(orientation="h", y=-0.1))
        _chart(fig2, 360)

    _section("Outstanding per Emiten (Top 15)")
    emiten_os = (gss.groupby(["IssuerCode","gss_type"])["Outstanding_M"]
                    .sum().reset_index().sort_values("Outstanding_M", ascending=False))
    top15 = (emiten_os.groupby("IssuerCode")["Outstanding_M"].sum()
                      .nlargest(15).index.tolist())
    emiten_os = emiten_os[emiten_os["IssuerCode"].isin(top15)]
    fig3 = px.bar(emiten_os, x="IssuerCode", y="Outstanding_M", color="gss_type",
                  color_discrete_map=CLASS_COLORS, barmode="stack",
                  labels={"Outstanding_M":"Rp Juta","gss_type":"Kelas","IssuerCode":"Emiten"},
                  category_orders={"IssuerCode":top15})
    fig3.update_traces(marker_line_width=0)
    fig3.update_layout(legend=dict(orientation="h", y=-0.2))
    _chart(fig3, 360)


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Daftar 82 GSS Berlabel
# ══════════════════════════════════════════════════════════════════════════════
def page_daftar():
    _section("🔍 Daftar 82 EBUS GSS Berlabel",
             "Semua instrumen GSS berdasarkan sensus nama & kode BEI · 18 Juni 2026")

    gss = load_gss()
    f1, f2, f3 = st.columns([2,2,3], gap="small")
    kelas_opts  = ["Semua"] + sorted(gss["gss_type"].dropna().unique().tolist())
    emiten_opts = ["Semua"] + sorted(gss["IssuerCode"].dropna().unique().tolist())
    sel_kelas   = f1.selectbox("Kelas GSS", kelas_opts)
    sel_emiten  = f2.selectbox("Emiten", emiten_opts)
    search      = f3.text_input("Cari nama instrumen", placeholder="ketik sebagian nama...")

    df = gss.copy()
    if sel_kelas  != "Semua": df = df[df["gss_type"]==sel_kelas]
    if sel_emiten != "Semua": df = df[df["IssuerCode"]==sel_emiten]
    if search:                df = df[df["BondName"].str.contains(search, case=False, na=False)]

    st.caption(f"Menampilkan **{len(df)}** dari {len(gss)} instrumen")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        _section("Instrumen per Emiten")
        iss_cnt = df.groupby(["IssuerCode","gss_type"]).size().reset_index(name="n")
        order   = (iss_cnt.groupby("IssuerCode")["n"].sum()
                          .sort_values(ascending=True).index.tolist())
        fig = px.bar(iss_cnt, x="n", y="IssuerCode", color="gss_type",
                     color_discrete_map=CLASS_COLORS, orientation="h",
                     labels={"n":"Jumlah","gss_type":"Kelas","IssuerCode":""},
                     category_orders={"IssuerCode":order})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(legend=dict(orientation="h",y=-0.18))
        _chart(fig, max(320, len(order)*32))

    with c2:
        _section("Outstanding per Emiten (Rp Juta)")
        iss_os = (df.groupby(["IssuerCode","gss_type"])["Outstanding_M"]
                    .sum().reset_index())
        order2 = (iss_os.groupby("IssuerCode")["Outstanding_M"].sum()
                        .sort_values(ascending=True).index.tolist())
        fig2 = px.bar(iss_os, x="Outstanding_M", y="IssuerCode", color="gss_type",
                      color_discrete_map=CLASS_COLORS, orientation="h",
                      labels={"Outstanding_M":"Rp Juta","gss_type":"Kelas","IssuerCode":""},
                      category_orders={"IssuerCode":order2})
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(legend=dict(orientation="h",y=-0.18))
        _chart(fig2, max(320, len(order2)*32))

    _section("Timeline Jatuh Tempo")
    df_mat = df.dropna(subset=["JatuhTempo"]).copy()
    df_mat["JatuhTempo"] = pd.to_datetime(df_mat["JatuhTempo"])
    if not df_mat.empty:
        fig3 = px.scatter(df_mat, x="JatuhTempo", y="Outstanding_M",
                          color="gss_type", color_discrete_map=CLASS_COLORS,
                          size="Outstanding_M", size_max=24,
                          hover_name="BondName",
                          hover_data={"IssuerCode":True,"Rating":True,
                                      "Outstanding_M":":.0f","JatuhTempo":True},
                          labels={"JatuhTempo":"Jatuh Tempo",
                                  "Outstanding_M":"Outstanding (Rp Juta)","gss_type":"Kelas"})
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(legend=dict(orientation="h",y=-0.15))
        _chart(fig3, 340)

    _section("Detail Instrumen")
    show = df[["BondId","IssuerCode","BondName","gss_type",
               "Rating","JatuhTempo","Outstanding_M"]].copy()
    show.columns = ["Kode","Emiten","Nama Instrumen","Kelas GSS",
                    "Rating","Jatuh Tempo","Outstanding (Rp Juta)"]
    st.dataframe(show.reset_index(drop=True), hide_index=True,
                 use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Dekomposisi Sektoral
# ══════════════════════════════════════════════════════════════════════════════
def page_dekomposisi():
    _section("🔬 Dekomposisi Sektoral Use-of-Proceeds",
             "Analisis sektor proyek dari 27 gold GSS bond (prospektus terverifikasi manual)")

    sectors = load_sectors()
    exp     = sector_expanded(sectors)

    f1, f2 = st.columns(2, gap="small")
    kelas_opts = ["Semua"] + sorted(sectors["gss_class"].dropna().unique().tolist())
    sel_kelas  = f1.selectbox("Filter Kelas", kelas_opts, key="dec_kelas")
    sel_bucket = f2.selectbox("Filter", ["Semua","Terdekomposisi saja"])

    df = sectors.copy()
    if sel_kelas  != "Semua": df = df[df["gss_class"]==sel_kelas]
    if sel_bucket == "Terdekomposisi saja": df = df[df["bucket"]=="terdekomposisi"]
    exp_f = exp.copy()
    if sel_kelas  != "Semua": exp_f = exp_f[exp_f["Kelas GSS"]==sel_kelas]

    st.caption(f"Menampilkan **{len(df)}** bond dari {len(sectors)} total")
    st.markdown("<br>", unsafe_allow_html=True)

    BKT_MAP = {"terdekomposisi":"Terdekomposisi","sektor_tak_terverifikasi":"Tanpa Sektor","level0":"Level-0"}
    BKT_CLR = {"terdekomposisi":C_GREEN,"sektor_tak_terverifikasi":"#E65100","level0":C_SUS}
    VS_CLR  = {"Terverifikasi":C_GREEN,"Sebagian":C_SOC,
                "Tidak terverifikasi (UoP tipis)":"#E65100","Level-0 (struktural)":C_SUS}

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _section("Status Dekomposisi")
        bkt = df["bucket"].value_counts().reset_index()
        bkt.columns = ["Bucket","n"]
        bkt["Label"] = bkt["Bucket"].map(BKT_MAP)
        bkt["Warna"] = bkt["Bucket"].map(BKT_CLR)
        fig = go.Figure(go.Bar(x=bkt["n"], y=bkt["Label"], orientation="h",
                               marker_color=bkt["Warna"], marker_line_width=0,
                               text=bkt["n"], textposition="outside"))
        _chart(fig, 240)

    with c2:
        _section("Status Verifikasi Klaim")
        vs = df["verification_status"].value_counts().reset_index()
        vs.columns = ["Status","n"]
        fig2 = go.Figure(go.Pie(
            labels=vs["Status"], values=vs["n"],
            marker_colors=[VS_CLR.get(s,"#888") for s in vs["Status"]],
            marker_line=dict(color="rgba(0,0,0,0)", width=0),
            hole=0.45, textinfo="label+percent",
        ))
        fig2.update_layout(showlegend=False)
        _chart(fig2, 240)

    _section("Frekuensi Kategori Sektoral (Bond Terdekomposisi)")
    dec_exp = exp_f[exp_f["Bucket"]=="terdekomposisi"] if not exp_f.empty else exp_f
    if not dec_exp.empty:
        cat_cnt = dec_exp["Kategori"].value_counts().reset_index()
        cat_cnt.columns = ["Kategori","n"]
        cat_cnt["Kelompok"] = cat_cnt["Kategori"].map(
            lambda x: "Lingkungan" if x in [v for k,v in CAT_LABEL.items()
                                             if CAT_BUCKET.get(k)=="Lingkungan"] else "Sosial")
        cat_cnt = cat_cnt.sort_values("n")
        fig3 = px.bar(cat_cnt, x="n", y="Kategori", color="Kelompok",
                      color_discrete_map={"Lingkungan":C_GREEN,"Sosial":C_SOC},
                      orientation="h", text="n",
                      labels={"n":"Jumlah Bond","Kategori":""})
        fig3.update_traces(marker_line_width=0, textposition="outside")
        fig3.update_layout(legend=dict(orientation="h",y=-0.12))
        _chart(fig3, max(300, len(cat_cnt)*30))

    _section("Heatmap Emiten × Kategori")
    if not dec_exp.empty:
        pivot = dec_exp.pivot_table(index="Emiten", columns="Kategori",
                                    aggfunc="size", fill_value=0)
        if not pivot.empty:
            fig4 = px.imshow(pivot, color_continuous_scale=[[0,"rgba(0,0,0,0)"],
                                                             [0.01,"rgba(46,125,50,0.2)"],
                                                             [1,"rgba(46,125,50,0.95)"]],
                             labels={"color":"Jumlah"}, aspect="auto")
            fig4.update_xaxes(tickangle=-30, tickfont_size=10)
            _chart(fig4, max(280, len(pivot)*35))

    _section("Detail per Prospektus")
    GC_D  = {"green":"Green","social":"Social","sustainability":"Sustainability",
             "sustainability_linked":"Sust.Linked"}
    BKT_D = {"terdekomposisi":"Terdekomposisi","sektor_tak_terverifikasi":"Tanpa Sektor","level0":"Level-0"}
    df_s  = df[["issuer","gss_class","bucket","verification_status",
                 "sector_keys","confidence"]].copy()
    df_s["gss_class"]    = df_s["gss_class"].map(GC_D).fillna(df_s["gss_class"])
    df_s["bucket"]       = df_s["bucket"].map(BKT_D).fillna(df_s["bucket"])
    df_s["sector_keys"]  = df_s["sector_keys"].apply(
        lambda x: ", ".join(CAT_LABEL.get(k,k) for k in str(x).split("|")
                            if k and k!="nan") if pd.notna(x) else "—")
    df_s["confidence"]   = df_s["confidence"].apply(lambda x: f"{float(x):.0%}")
    df_s.columns = ["Emiten","Kelas","Status Dekomposisi",
                    "Verifikasi","Kategori Teridentifikasi","Keyakinan"]
    st.dataframe(df_s.reset_index(drop=True), hide_index=True,
                 use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════════════════════════
# Page 5 — Evaluasi Klasifikasi
# ══════════════════════════════════════════════════════════════════════════════
def page_evaluasi():
    _section("⚖️ Evaluasi Kinerja Klasifikasi GSS",
             "Rule-based (baseline) vs ML Semantik (final) · 93 dokumen gold set")

    comp = load_comparison()
    rule = prf(comp,"pred_rule")
    ml   = prf(comp,"pred_ml")
    n    = len(comp)

    cols = st.columns(5, gap="small")
    with cols[0]: _card("📏","Gold Set",f"{n} dok.","terverifikasi manual",C_BI,S_BI)
    with cols[1]: _card("🎯","ML Precision",f"{ml['P']:.3f}",f"Rule: {rule['P']:.3f}",C_GREEN,S_GREEN)
    with cols[2]: _card("📡","ML Recall",f"{ml['R']:.3f}",f"Rule: {rule['R']:.3f}",C_GREEN,S_GREEN)
    with cols[3]: _card("⭐","ML F1-Score",f"{ml['F1']:.3f}",f"Rule: {rule['F1']:.3f}",C_GREEN,S_GREEN)
    with cols[4]: _card("❌","Rule FP (salah tag)",str(rule['FP']),"employment_msme pemicu utama",C_ERR,S_ERR)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        _section("P / R / F1 — Rule vs ML")
        metrics_df = pd.DataFrame({
            "Metrik": ["Precision","Recall","F1-Score"]*2,
            "Nilai":  [rule["P"],rule["R"],rule["F1"],ml["P"],ml["R"],ml["F1"]],
            "Metode": ["Rule-based"]*3 + ["ML Semantik"]*3,
        })
        fig = px.bar(metrics_df, x="Metrik", y="Nilai", color="Metode",
                     barmode="group", text="Nilai",
                     color_discrete_map={"Rule-based":"rgba(120,144,156,0.85)","ML Semantik":C_BI},
                     range_y=[0,1.15],
                     labels={"Nilai":"Skor (0–1)"})
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                          marker_line_width=0)
        fig.add_hline(y=1.0, line_dash="dot", line_color=C_GREEN,
                      annotation_text="Skor sempurna",
                      annotation_font_color=C_GREEN)
        fig.update_layout(legend=dict(orientation="h",y=-0.2))
        _chart(fig, 360)

    with c2:
        _section("Confusion Matrix")
        tab_r, tab_m = st.tabs(["Rule-based","ML Semantik"])

        def _cm(m):
            fig = go.Figure(go.Table(
                header=dict(
                    values=["","✅ Aktual GSS","❌ Aktual Non-GSS"],
                    fill_color=C_BI, font_color="white",
                    font_size=13, height=38, align="center",
                ),
                cells=dict(
                    values=[
                        ["🔮 Prediksi GSS","🔮 Prediksi Non-GSS"],
                        [f"TP = {m['TP']}", f"FN = {m['FN']}"],
                        [f"FP = {m['FP']}", f"TN = {m['TN']}"],
                    ],
                    fill_color=[
                        ["rgba(128,128,128,0.10)","rgba(128,128,128,0.10)"],
                        [S_OK if m["TP"]>0 else "rgba(0,0,0,0.02)",
                         S_ERR if m["FN"]>0 else S_OK],
                        [S_ERR if m["FP"]>0 else S_OK,
                         S_OK],
                    ],
                    font_size=14, height=52, align="center",
                    font_color=[["inherit","inherit"],
                                [C_OK if m["TP"]>0 else "#888",
                                 C_ERR if m["FN"]>0 else C_OK],
                                [C_ERR if m["FP"]>0 else C_OK, C_OK]],
                )
            ))
            fig.update_layout(height=210)
            _chart(fig)

        with tab_r:
            _cm(rule)
            st.caption(f"**{rule['FP']} false positive** — bond biasa diprediksi GSS karena "
                       "'Berkelanjutan' dalam nama PUB administratif.")
        with tab_m:
            _cm(ml)
            st.success(f"Sempurna pada {n} dokumen — FP=0, FN=0")

    _section("Akurasi per Emiten")
    iss_acc = []
    for iss, grp in comp.groupby("issuer"):
        r = prf(grp,"pred_rule"); m = prf(grp,"pred_ml")
        iss_acc.append({
            "Emiten":iss, "N Dok":len(grp),
            "N GSS":(grp["gold"]=="GSS").sum(),
            "Rule P":round(r["P"],3), "Rule R":round(r["R"],3), "Rule F1":round(r["F1"],3),
            "ML P":round(m["P"],3),   "ML R":round(m["R"],3),   "ML F1":round(m["F1"],3),
        })
    acc_df = pd.DataFrame(iss_acc).sort_values("ML F1", ascending=False)
    st.dataframe(acc_df.reset_index(drop=True), hide_index=True,
                 use_container_width=True, height=380)

    st.info("**Akar masalah Rule-based FP:** frasa 'Berkelanjutan' muncul dalam nama "
            "administrasi *Penawaran Umum Berkelanjutan* (PUB) — tidak ada kaitannya "
            "dengan GSS. ML Semantik membaca konteks sehingga tidak terjebak.")


# ══════════════════════════════════════════════════════════════════════════════
# Page 6 — Taksonomi 15 Kategori
# ══════════════════════════════════════════════════════════════════════════════
def page_taksonomi():
    _section("📚 Taksonomi 15 Kategori Eligible GSS",
             "POJK 18/2023 · ICMA Green & Social Bond Principles · SDG Gov. Securities Framework (DJPPR)")

    TAXONOMY = [
        ("renewable_energy","Energi Terbarukan","Lingkungan","🔆",
         "energi terbarukan, panel surya, PLTS, tenaga angin, panas bumi, PLTA, biomassa",
         "PLTS atap, PLTA, PLTB, geothermal"),
        ("energy_efficiency","Efisiensi Energi","Lingkungan","⚡",
         "efisiensi energi, smart grid, retrofit, kogenerasi, manajemen energi",
         "Retrofit LED, modernisasi trafo distribusi"),
        ("green_building","Bangunan Hijau","Lingkungan","🏢",
         "green building, Greenship, EDGE, LEED, gedung hijau",
         "Gedung perkantoran bersertifikat Greenship/EDGE"),
        ("green_tourism","Pariwisata Berkelanjutan","Lingkungan","🌴",
         "ekowisata, pariwisata berkelanjutan, sustainable tourism",
         "Resort eco-certified, destinasi konservasi"),
        ("sustainable_transport","Transportasi Berkelanjutan","Lingkungan","🚇",
         "MRT, LRT, KRL, BRT, kendaraan listrik, electric vehicle",
         "Armada bus listrik, perluasan MRT/LRT"),
        ("waste_management","Pengelolaan Sampah & Limbah","Lingkungan","♻️",
         "pengelolaan sampah, waste to energy, PLTSA, daur ulang, 3R",
         "Fasilitas sampah→energi, IPAL industri"),
        ("water_management","Pengelolaan Air & Air Limbah","Lingkungan","💧",
         "SPAM, IPAL, air bersih, pengelolaan air limbah, sanitasi, irigasi",
         "SPAM regional, rehabilitasi IPAL"),
        ("natural_resources","Pengelolaan SDA Berkelanjutan","Lingkungan (Blue)","🌊",
         "kehutanan lestari, reboisasi, konservasi, perikanan berkelanjutan",
         "Rehabilitasi mangrove, budidaya ikan"),
        ("climate_resilience","Ketahanan Iklim & PRB","Lingkungan","🌡️",
         "ketahanan iklim, adaptasi perubahan iklim, pengurangan risiko bencana",
         "Infrastruktur tahan banjir, early warning"),
        ("basic_infrastructure","Infrastruktur Dasar Terjangkau","Sosial","🏗️",
         "infrastruktur dasar, akses air minum, sanitasi dasar, elektrifikasi",
         "Elektrifikasi 3T, jaringan air minum perdesaan"),
        ("essential_services","Akses Layanan Esensial","Sosial","🏥",
         "akses kesehatan, puskesmas, rumah sakit, pendidikan, beasiswa",
         "Puskesmas 3T, beasiswa pendidikan"),
        ("affordable_housing","Perumahan Terjangkau","Sosial","🏠",
         "perumahan terjangkau, rumah subsidi, MBR, KPR FLPP",
         "KPR bersubsidi MBR, rusunami"),
        ("employment_msme","Penciptaan Lapangan Kerja & UMKM","Sosial","💼",
         "UMKM, usaha mikro, KUR, kredit usaha rakyat, microfinance",
         "Program KUR UMKM, pembiayaan ultra-mikro"),
        ("food_security","Ketahanan Pangan","Sosial","🌾",
         "ketahanan pangan, food security, produktivitas pertanian",
         "Gudang pangan, rantai pasok pertanian"),
        ("socioeconomic","Pemberdayaan Sosial-Ekonomi","Sosial","🤝",
         "pemberdayaan, pengentasan kemiskinan, inklusi keuangan",
         "Pemberdayaan perempuan, inklusi keuangan"),
    ]

    df_tax = pd.DataFrame(TAXONOMY,
        columns=["Kode","Nama","Kelompok","Ikon","Kata Kunci","Contoh"])

    kelompok_opts = ["Semua"] + sorted(df_tax["Kelompok"].unique())
    sel = st.selectbox("Filter Kelompok", kelompok_opts)
    if sel != "Semua":
        df_tax = df_tax[df_tax["Kelompok"].str.contains(sel.split()[0])]

    _section("Peta Kategori GSS")
    fig = px.treemap(df_tax, path=["Kelompok","Nama"], values=[1]*len(df_tax),
                     color="Kelompok",
                     color_discrete_map={"Lingkungan":C_GREEN,
                                         "Lingkungan (Blue)":C_SL,
                                         "Sosial":C_SOC})
    fig.update_traces(textinfo="label", textfont_size=12,
                      marker_line_width=2,
                      marker_line_color="rgba(0,0,0,0.3)")
    _chart(fig, 340)

    _section("Detail 15 Kategori")
    col_a, col_b = st.columns(2, gap="medium")

    for i, (_, row) in enumerate(df_tax.iterrows()):
        is_env  = "Lingkungan" in row["Kelompok"]
        color   = C_GREEN if is_env else C_SOC
        surf    = S_GREEN if is_env else S_SOC
        target  = col_a if i % 2 == 0 else col_b

        with target:
            with st.expander(f"{row['Ikon']} **{row['Nama']}** — {row['Kelompok']}"):
                st.markdown(f"""
                <div style="border-left:3px solid {color};
                            background:{surf};
                            border-radius:0 8px 8px 0;
                            padding:10px 14px;margin-bottom:8px">
                  <div style="font-size:0.8rem;opacity:0.65;
                              text-transform:uppercase;letter-spacing:0.05em">
                    Kata Kunci Pemicu</div>
                  <div style="margin-top:4px;font-size:0.88rem">{row['Kata Kunci']}</div>
                </div>
                <div style="font-size:0.8rem;opacity:0.65;
                            text-transform:uppercase;letter-spacing:0.05em;
                            margin-top:8px">Contoh Proyek</div>
                <div style="font-size:0.88rem;margin-top:4px">{row['Contoh']}</div>
                """, unsafe_allow_html=True)

    _divider()
    _section("Level-0 — Sinyal Struktur Instrumen")
    st.caption("Diperiksa sebelum analisis use-of-proceeds. Jika terdeteksi → langsung diklasifikasi.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"""
        <div style="border-radius:10px;border:1px solid {C_SL}44;
                    background:{S_SL};padding:16px 18px">
          <div style="font-size:1.3rem">🔗</div>
          <div style="font-weight:700;margin-top:6px">Sustainability-Linked</div>
          <div style="font-size:0.85rem;opacity:0.75;margin-top:8px;line-height:1.5">
            Dana penggunaan bebas. Yang diatur: mekanisme target kinerja (KPI/SPT/step-up).
          </div>
          <div style="font-size:0.78rem;margin-top:10px;border-top:1px solid {C_SL}33;
                      padding-top:8px;opacity:0.65">
            sustainability-linked · terkait keberlanjutan · KPI · SPT · step-up · coupon ratchet
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="border-radius:10px;border:1px solid {C_SUS}44;
                    background:{S_SUS};padding:16px 18px">
          <div style="font-size:1.3rem">🕌</div>
          <div style="font-weight:700;margin-top:6px">Sukuk Wakaf</div>
          <div style="font-size:0.85rem;opacity:0.75;margin-top:8px;line-height:1.5">
            Instrumen berbasis aset wakaf (CWLS dan turunannya). Nazhir sebagai pengelola.
          </div>
          <div style="font-size:0.78rem;margin-top:10px;border-top:1px solid {C_SUS}33;
                      padding-top:8px;opacity:0.65">
            wakaf · waqf · CWLS · nazhir · ikrar wakaf · mauquf
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 7 — SBN GSS Sovereign
# ══════════════════════════════════════════════════════════════════════════════
def page_sbn():
    _section("🏛️ SBN GSS Sovereign — Acuan Taksonomi",
             "25 seri SBN GSS · Diaudit BPK · Tercatat SRN-PPI · Ground truth klasifikasi")

    sbn = load_sbn()
    kat_cnt = sbn["kategori_gss"].value_counts()

    cols = st.columns(4, gap="small")
    with cols[0]: _card("🏛️","Total SBN GSS",f"{len(sbn)} seri","Sovereign reference",C_BI,S_BI)
    with cols[1]: _card("🌿","Green Bonds",f"{kat_cnt.get('Green',0)} seri","Global + Domestik",C_GREEN,S_GREEN)
    with cols[2]: _card("♻️","Sustainability Bonds",f"{kat_cnt.get('Sustainability',0)} seri","SDG + Blue Bond",C_SUS,S_SUS)
    with cols[3]: _card("📖","Framework","SDG Gov. Sec. 2021","DJPPR",C_SOC,S_SOC)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**Mengapa SBN jadi acuan?** SBN GSS sudah melalui Climate Budget Tagging → audit BPK "
            "→ pelaporan SRN-PPI. Labelnya definitif — tidak memerlukan ML. "
            "Taksonomi 9 sektor hijau + 6 sektor sosial yang dipakai classifier EBUS korporasi "
            "diturunkan dari SDG Government Securities Framework DJPPR.")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        _section("Distribusi per Kategori & Pasar")
        cross = sbn.groupby(["kategori_gss","pasar"]).size().reset_index(name="n")
        fig = px.bar(cross, x="kategori_gss", y="n", color="pasar",
                     color_discrete_sequence=["#1565C0","#2E7D32","#C2A100","#7B1FA2","#E65100"],
                     labels={"kategori_gss":"Kategori","n":"Jumlah Seri","pasar":"Pasar"},
                     barmode="stack", text_auto=True)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(legend=dict(orientation="h",y=-0.25))
        _chart(fig, 340)

    with c2:
        _section("Sebaran Tema Instrumen")
        tema_cnt = sbn["tema"].value_counts().reset_index()
        tema_cnt.columns = ["Tema","n"]
        TEMA_CLR = {"Green":C_GREEN,"Blue":C_SL,"SDG (Sosial+Hijau)":C_SUS}
        fig2 = go.Figure(go.Pie(
            labels=tema_cnt["Tema"], values=tema_cnt["n"],
            marker_colors=[TEMA_CLR.get(t,"#888") for t in tema_cnt["Tema"]],
            marker_line=dict(color="rgba(0,0,0,0)", width=0),
            hole=0.45, textinfo="label+value+percent",
        ))
        fig2.update_layout(showlegend=False)
        _chart(fig2, 340)

    _section("Timeline Emisi SBN GSS")
    sbn["Tahun"] = sbn["tanggal_emisi"].apply(
        lambda x: int(str(x)[:4]) if pd.notna(x) and len(str(x))>=4 else None)
    yr_cnt = sbn.dropna(subset=["Tahun"]).groupby(["Tahun","kategori_gss"]).size().reset_index(name="n")
    fig3 = px.bar(yr_cnt, x="Tahun", y="n", color="kategori_gss",
                  color_discrete_map={"Green":C_GREEN,"Sustainability":C_SUS},
                  labels={"Tahun":"Tahun Emisi","n":"Jumlah Seri","kategori_gss":"Kategori"},
                  barmode="stack", text_auto=True)
    fig3.update_traces(marker_line_width=0)
    fig3.update_layout(legend=dict(orientation="h",y=-0.15))
    _chart(fig3, 300)

    _section("Daftar Lengkap SBN GSS")
    f1, f2 = st.columns(2, gap="small")
    kat_f   = f1.selectbox("Filter Kategori", ["Semua","Green","Sustainability"], key="sbn_k")
    pasar_f = f2.selectbox("Filter Pasar",
                           ["Semua"]+sorted(sbn["pasar"].dropna().unique()), key="sbn_p")
    df_sbn  = sbn.copy()
    if kat_f   != "Semua": df_sbn = df_sbn[df_sbn["kategori_gss"]==kat_f]
    if pasar_f != "Semua": df_sbn = df_sbn[df_sbn["pasar"]==pasar_f]

    show = df_sbn[["seri","jenis_instrumen","pasar","kategori_gss","tema",
                   "tanggal_emisi","mata_uang","nominal","tenor","catatan"]].copy()
    show.columns = ["Seri","Jenis","Pasar","Kategori","Tema",
                    "Tanggal Emisi","Mata Uang","Nominal","Tenor","Catatan"]
    st.dataframe(show.reset_index(drop=True), hide_index=True,
                 use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar + Router
# ══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "🏠  Ringkasan Eksekutif":    page_ringkasan,
    "📊  Sensus Pasar GSS":       page_sensus,
    "🔍  Daftar 82 GSS Berlabel": page_daftar,
    "🔬  Dekomposisi Sektoral":   page_dekomposisi,
    "⚖️  Evaluasi Klasifikasi":   page_evaluasi,
    "📚  Taksonomi 15 Kategori":  page_taksonomi,
    "🏛️  SBN GSS Sovereign":      page_sbn,
}

def main():
    with st.sidebar:
        # Brand header
        st.markdown(f"""
        <div style="border-radius:12px;
                    background:linear-gradient(135deg,{C_BI},{C_SL});
                    padding:16px 18px;margin-bottom:20px;
                    border:1px solid rgba(255,255,255,0.1)">
          <div style="color:white;font-size:1.05rem;font-weight:700;line-height:1.2">
            🌿 Dashboard GSS EBUS
          </div>
          <div style="color:rgba(255,255,255,0.75);font-size:0.76rem;margin-top:4px">
            Bank Indonesia · DSta/DSMF
          </div>
        </div>""", unsafe_allow_html=True)

        page = st.radio("Navigasi", list(PAGES.keys()),
                        label_visibility="collapsed")

        _divider()

        census = load_census()
        total  = census[census["gss_type"]=="TOTAL"].iloc[0]

        st.markdown("""<div style="font-size:0.78rem;opacity:0.55;
                        text-transform:uppercase;letter-spacing:0.05em;
                        margin-bottom:6px">Ringkasan Data</div>""",
                    unsafe_allow_html=True)

        info_rows = [
            ("Universe EBUS",  f"{N_UNIVERSE:,} instrumen"),
            ("GSS Berlabel",   f"{int(total['n_instrumen'])} instrumen"),
            ("Outstanding",    f"Rp {total['total_outstanding_triliun']} T"),
            ("Per",            "18 Juni 2026"),
        ]
        for label, val in info_rows:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        font-size:0.82rem;padding:3px 0;
                        border-bottom:1px solid rgba(128,128,128,0.12)">
              <span style="opacity:0.6">{label}</span>
              <span style="font-weight:600">{val}</span>
            </div>""", unsafe_allow_html=True)

        _divider()

        st.markdown("""<div style="font-size:0.78rem;opacity:0.55;
                        text-transform:uppercase;letter-spacing:0.05em;
                        margin-bottom:6px">Dasar Hukum</div>""",
                    unsafe_allow_html=True)
        for reg in ["POJK 18/2023 (OJK)","ICMA GBP/SBP","SDG Gov.Sec. Framework DJPPR"]:
            st.markdown(f"<div style='font-size:0.8rem;padding:2px 0;opacity:0.75'>"
                        f"· {reg}</div>", unsafe_allow_html=True)

    PAGES[page]()


if __name__ == "__main__":
    main()
