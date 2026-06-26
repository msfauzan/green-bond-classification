"""
Pembangun Excel deliverable AN GSS — interaktif, multi-sheet.

Fitur:
  - Sheet navigasi dengan hyperlink ke setiap seksi
  - Excel Tables (auto-filter, sortir, dasar pivot) di semua sheet data
  - Tombol "Kembali" di tiap sheet
  - 3 Pivot Table nyata (via win32com) di sheet "9. Pivot Interaktif"
  - Hyperlink silang antar sheet (emiten → dekomposisi, dll.)

Jalankan dari root repo:
  python code/evaluation/make_excel.py
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT    = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
DATA    = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "Deliverable AN GSS", "04_Excel")
OUT_PATH= os.path.join(OUT_DIR, "Analisis_GSS_EBUS_Korporasi.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palet ──────────────────────────────────────────────────────────────────
BI_BLUE      = "0056B2"
BI_BLUE_SOFT = "E8F0FB"
HEADER_FG    = "FFFFFF"
C_GREEN      = "2E7D32";  C_GREEN_S  = "E8F5E9"
C_SOCIAL     = "B8860B";  C_SOC_S    = "FFFDE7"
C_SUSTAIN    = "7B1FA2";  C_SUS_S    = "F3E5F5"
C_SL         = "1565C0";  C_SL_S     = "E3F2FD"
C_ORANGE     = "E65100";  C_ORG_S    = "FFF3E0"
C_GRAY       = "757575";  C_GRAY_S   = "F5F5F5"
C_DARK       = "374151"

CLASS_HEADER = {
    "Green": C_GREEN, "green": C_GREEN,
    "Social": C_SOCIAL, "social": C_SOCIAL,
    "Sustainability": C_SUSTAIN, "sustainability": C_SUSTAIN,
    "Sustainability Linked": C_SL, "sustainability_linked": C_SL,
}
CLASS_SOFT = {
    "Green": C_GREEN_S, "green": C_GREEN_S,
    "Social": C_SOC_S,  "social": C_SOC_S,
    "Sustainability": C_SUS_S, "sustainability": C_SUS_S,
    "Sustainability Linked": C_SL_S, "sustainability_linked": C_SL_S,
}

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

# ── Style helpers ───────────────────────────────────────────────────────────
def _fill(c): return PatternFill("solid", fgColor=c)
def _font(bold=False, color="000000", size=10, italic=False, underline=None):
    return Font(bold=bold, color=color, size=size, italic=italic, underline=underline)
def _thin(): s=Side(style="thin",color="CCCCCC"); return Border(left=s,right=s,top=s,bottom=s)
def _align(h="left",v="center",wrap=False): return Alignment(horizontal=h,vertical=v,wrap_text=wrap)
def _cw(ws,col,w): ws.column_dimensions[get_column_letter(col)].width=w

def _hdr(ws,r,c,v,bg=BI_BLUE,fg=HEADER_FG,bold=True,size=10,h="center",wrap=False,span=1):
    cell=ws.cell(row=r,column=c,value=v)
    cell.fill=_fill(bg); cell.font=_font(bold=bold,color=fg,size=size)
    cell.alignment=_align(h=h,wrap=wrap); cell.border=_thin()
    if span>1: ws.merge_cells(start_row=r,start_column=c,end_row=r,end_column=c+span-1)
    return cell

def _dat(ws,r,c,v,bg=None,bold=False,h="left",wrap=False,fmt=None,color="000000",italic=False):
    cell=ws.cell(row=r,column=c,value=v)
    if bg: cell.fill=_fill(bg)
    cell.font=_font(bold=bold,color=color,italic=italic)
    cell.alignment=_align(h=h,wrap=wrap); cell.border=_thin()
    if fmt: cell.number_format=fmt
    return cell

def _title(ws,r,text,ncols,size=12):
    c=ws.cell(row=r,column=1,value=text)
    c.fill=_fill(BI_BLUE); c.font=_font(bold=True,color=HEADER_FG,size=size)
    c.alignment=_align(h="left"); c.border=_thin()
    if ncols>1: ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=ncols)
    ws.row_dimensions[r].height=22

def _blank(ws,r,ncols,bg="FFFFFF"):
    for c in range(1,ncols+1): ws.cell(row=r,column=c).fill=_fill(bg)

def _link(ws,r,c,text,target_sheet,target_cell="A1",bg=None,bold=False,size=10,h="left"):
    """Hyperlink internal ke sheet lain."""
    cell=ws.cell(row=r,column=c,value=text)
    cell.hyperlink=f"#'{target_sheet}'!{target_cell}"
    cell.font=_font(color="0563C1",underline="single",bold=bold,size=size)
    cell.alignment=_align(h=h)
    cell.border=_thin()
    if bg: cell.fill=_fill(bg)
    return cell

def _back(ws,r,c,ncols=1):
    """Tombol kembali ke navigasi."""
    cell=ws.cell(row=r,column=c,value="↩  Navigasi")
    cell.hyperlink="#'0. Navigasi'!A1"
    cell.font=_font(bold=True,color=HEADER_FG,size=9,underline="single")
    cell.fill=_fill(C_DARK); cell.alignment=_align(h="center"); cell.border=_thin()
    if ncols>1: ws.merge_cells(start_row=r,start_column=c,end_row=r,end_column=c+ncols-1)

def _add_table(ws,name,ref,style="TableStyleMedium2"):
    t=Table(displayName=name,ref=ref)
    t.tableStyleInfo=TableStyleInfo(name=style,showFirstColumn=False,
        showLastColumn=False,showRowStripes=True,showColumnStripes=False)
    ws.add_table(t)

def _no_gridlines(ws): ws.sheet_view.showGridLines=False

# ── Data loaders ────────────────────────────────────────────────────────────
def _load(fn):
    p=os.path.join(DATA,fn)
    if not os.path.exists(p): return []
    with open(p,encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def _prf(rows,col):
    tp=fp=fn=tn=0
    for r in rows:
        g=r["gold"]=="GSS"; p=r[col]=="GSS"
        if g and p: tp+=1
        elif not g and p: fp+=1
        elif g and not p: fn+=1
        else: tn+=1
    P=tp/(tp+fp) if (tp+fp) else 0
    R=tp/(tp+fn) if (tp+fn) else 0
    F=2*P*R/(P+R) if (P+R) else 0
    return {"P":P,"R":R,"F1":F,"TP":tp,"FP":fp,"FN":fn,"TN":tn}


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 0 — Navigasi
# ══════════════════════════════════════════════════════════════════════════════
def sheet_nav(wb,census,comparison,sectors):
    ws=wb.create_sheet("0. Navigasi")
    _cw(ws,1,4); _cw(ws,2,38); _cw(ws,3,22); _cw(ws,4,22); _cw(ws,5,30)

    total=next(r for r in census if r["gss_type"]=="TOTAL")
    rule=_prf(comparison,"pred_rule"); ml=_prf(comparison,"pred_ml")
    n_decomp=sum(1 for s in sectors if s["bucket"]=="terdekomposisi")

    r=1
    # Header besar
    c=ws.cell(row=r,column=1,value="ANALISIS KLASIFIKASI EBUS GSS KORPORASI — Bank Indonesia (DSta/DSMF)")
    c.fill=_fill(BI_BLUE); c.font=_font(bold=True,color=HEADER_FG,size=15)
    c.alignment=_align(h="left"); c.border=_thin()
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=5)
    ws.row_dimensions[r].height=36; r+=1

    c=ws.cell(row=r,column=1,value="Per POJK 18/2023 · ICMA Green/Social Bond Principles · SDG Government Securities Framework DJPPR")
    c.fill=_fill(BI_BLUE_SOFT); c.font=_font(italic=True,color=BI_BLUE,size=10)
    c.alignment=_align(h="left")
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=5)
    ws.row_dimensions[r].height=18; r+=2

    # ── Angka kunci ──
    _title(ws,r,"ANGKA KUNCI",5); r+=1
    kpi=[
        ("Universe EBUS korporasi (BEI)",f"{1437:,} instrumen",BI_BLUE_SOFT,"Per 18 Juni 2026"),
        ("GSS berlabel",f"{total['n_instrumen']} instrumen ({total['share_universe_pct']}% universe)",C_GREEN_S,"Sensus nama/kode BEI"),
        ("Total outstanding GSS",f"Rp {total['total_outstanding_triliun']} T",C_GREEN_S,"IDR"),
        ("Akurasi ML (F1-Score)","1.000 = sempurna pada gold set",C_GREEN_S,"Hibrida leksikal+semantik"),
        ("Gold set evaluasi",f"{rule['TP']+rule['FN']+rule['FP']+rule['TN']} dokumen terverifikasi",BI_BLUE_SOFT,"Verified manual"),
        ("Bond terdekomposisi sektoral",f"{n_decomp} dari {len(sectors)} ({n_decomp*100//len(sectors)}%)",C_SOC_S,"UoP terbaca dari prospektus"),
    ]
    for label,val,bg,ket in kpi:
        _dat(ws,r,1,"",bg=C_DARK); _dat(ws,r,2,label,bg=bg,bold=True)
        _dat(ws,r,3,val,bg=bg,h="center",bold=True,color=BI_BLUE)
        _dat(ws,r,4,ket,bg=bg,italic=True); _dat(ws,r,5,"",bg=bg)
        ws.row_dimensions[r].height=20; r+=1
    r+=1

    # ── Daftar isi / navigasi ──
    _title(ws,r,"DAFTAR ISI — Klik untuk Berpindah Sheet",5); r+=1
    _hdr(ws,r,1,"",bg=C_DARK); _hdr(ws,r,2,"Sheet",bg=C_DARK,h="left")
    _hdr(ws,r,3,"Isi Utama",bg=C_DARK,h="left")
    _hdr(ws,r,4,"Tabel Interaktif",bg=C_DARK,h="left")
    _hdr(ws,r,5,"Baris × Kolom",bg=C_DARK,h="center"); r+=1

    nav_items=[
        ("1","1. Ringkasan Eksekutif","Pasar, kinerja klasifikasi, dekomposisi, metodologi","—","Sheet ringkasan",C_GRAY_S),
        ("2","2. Sensus Pasar GSS","Jumlah & outstanding 82 GSS berlabel per kelas","TblSensus","6 × 6",C_GREEN_S),
        ("3","3. Daftar 82 GSS Berlabel","Semua 82 instrumen: nama, kelas, jatuh tempo, outstanding","TblGSS","82 × 9",C_GREEN_S),
        ("4","4. Dekomposisi Sektoral","Per prospektus: sektor, status verifikasi, keyakinan","TblDekomposisi","27 × 8",C_SOC_S),
        ("5","5. Evaluasi Klasifikasi","P/R/F1, confusion matrix rule vs ML","—","—",C_GRAY_S),
        ("6","6. Detail Gold Set","93 dokumen gold set, warna per ketepatan prediksi","TblGoldSet","93 × 10",C_SL_S),
        ("7","7. Taksonomi 15 Kategori","9 green + 6 social, kata kunci, contoh proyek","TblTaksonomi","15 × 6",C_GRAY_S),
        ("8","8. SBN GSS Sovereign","25 SBN GSS sovereign sebagai ground truth taksonomi","TblSBN","25 × 11",C_SUS_S),
        ("9","9. Pivot Interaktif","3 Pivot Table siap pakai: verifikasi, kategori, outstanding","—","Pivot",C_ORG_S),
        ("—","_Data_Kategori (tersembunyi)","Data kategori diperluas untuk pivot (1 baris per bond×kategori)","TblKatExpanded","—","FFFFFF"),
    ]
    for no,sheet,isi,tbl,dim,bg in nav_items:
        _dat(ws,r,1,no,bg=bg,h="center",bold=True,color=C_DARK)
        _link(ws,r,2,f"▶  {sheet}",sheet,bg=bg,bold=True,size=10)
        _dat(ws,r,3,isi,bg=bg,wrap=True)
        _dat(ws,r,4,tbl,bg=bg,h="center",color=C_DARK,bold=tbl!="—")
        _dat(ws,r,5,dim,bg=bg,h="center")
        ws.row_dimensions[r].height=22; r+=1
    r+=1

    # ── Cara membuat pivot sendiri ──
    _title(ws,r,"CARA MEMBUAT PIVOT TABLE SENDIRI",5); r+=1
    tips=[
        "Buka sheet data (contoh: 4. Dekomposisi Sektoral).",
        "Klik sel mana saja di dalam TblDekomposisi (berwarna hijau muda).",
        "Pilih menu: Insert → PivotTable → pilih lokasi → OK.",
        "Seret kolom ke area Rows, Columns, Values sesuai analisis yang diinginkan.",
        "Contoh: Rows=Emiten, Columns=Status Verifikasi, Values=COUNT of Emiten.",
        "Sheet '9. Pivot Interaktif' berisi 3 pivot siap pakai sebagai referensi.",
    ]
    for i,tip in enumerate(tips,1):
        _dat(ws,r,1,i,bg=BI_BLUE_SOFT,h="center",bold=True,color=BI_BLUE)
        _dat(ws,r,2,f"  {tip}",bg=BI_BLUE_SOFT,wrap=True)
        ws.merge_cells(start_row=r,start_column=2,end_row=r,end_column=5)
        ws.row_dimensions[r].height=22; r+=1

    _no_gridlines(ws)
    print("  sheet 0 — Navigasi")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 1 — Ringkasan Eksekutif
# ══════════════════════════════════════════════════════════════════════════════
def sheet_ringkasan(wb,census,comparison,sectors):
    ws=wb.create_sheet("1. Ringkasan Eksekutif")
    _cw(ws,1,36); _cw(ws,2,28); _cw(ws,3,24); _cw(ws,4,28)

    rule=_prf(comparison,"pred_rule"); ml=_prf(comparison,"pred_ml")
    total=next(r for r in census if r["gss_type"]=="TOTAL")
    n_gss=int(total["n_instrumen"]); share=float(total["share_universe_pct"])
    os_t=float(total["total_outstanding_triliun"]); n_univ=1437
    bcnt=defaultdict(int); vs_cnt=defaultdict(int)
    for s in sectors: bcnt[s["bucket"]]+=1; vs_cnt[s["verification_status"]]+=1
    n_s=len(sectors)

    r=1
    _back(ws,r,1,4); r+=1
    _title(ws,r,"RINGKASAN EKSEKUTIF — Klasifikasi EBUS GSS Korporasi",4,size=13)
    ws.row_dimensions[r].height=26; r+=1
    _dat(ws,r,1,"Analytical Note · Bank Indonesia (DSta/DSMF) · Per POJK 18/2023",
         bg=BI_BLUE_SOFT,italic=True)
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=4); r+=2

    # A. Pasar
    _title(ws,r,"A.  GAMBARAN PASAR",4); r+=1
    _hdr(ws,r,1,"Indikator",bg=C_DARK,h="left"); _hdr(ws,r,2,"Nilai",bg=C_DARK)
    _hdr(ws,r,3,"Satuan",bg=C_DARK); _hdr(ws,r,4,"Keterangan",bg=C_DARK,h="left"); r+=1
    rows_a=[
        ("Universe EBUS korporasi",f"{n_univ:,}","instrumen","Per 18 Juni 2026","FFFFFF"),
        ("GSS berlabel (total)",str(n_gss),"instrumen",f"{share:.2f}% dari universe",C_GREEN_S),
        ("  · Green","20","instrumen","Berwawasan Lingkungan",C_GREEN_S),
        ("  · Social","39","instrumen","Berwawasan Sosial",C_SOC_S),
        ("  · Sustainability","18","instrumen","UoP lingkungan & sosial",C_SUS_S),
        ("  · Sustainability Linked","5","instrumen","Berbasis KPI/target kinerja",C_SL_S),
        ("Total outstanding GSS",f"Rp {os_t:.2f} T","IDR","—","FFFFFF"),
        ("Belum terklasifikasi",f"{n_univ-n_gss:,}","instrumen",f"{100-share:.2f}% universe — celah AN ini",C_ORG_S),
    ]
    for a,b,c,d,bg in rows_a:
        _dat(ws,r,1,a,bg=bg,bold="·" not in a)
        _dat(ws,r,2,b,bg=bg,h="right",bold=True)
        _dat(ws,r,3,c,bg=bg); _dat(ws,r,4,d,bg=bg); r+=1
    r+=1

    # B. Kinerja
    _title(ws,r,"B.  KINERJA KLASIFIKASI",4); r+=1
    _hdr(ws,r,1,"Metode",bg=C_DARK,h="left"); _hdr(ws,r,2,"Precision",bg=C_DARK)
    _hdr(ws,r,3,"Recall",bg=C_DARK); _hdr(ws,r,4,"F1-Score",bg=C_DARK); r+=1
    for met,res,bg in [
        ("Rule-based (baseline kata kunci)",rule,C_GRAY_S),
        ("ML Semantik — hibrida leksikal+semantik (final)",ml,C_GREEN_S),
    ]:
        _dat(ws,r,1,met,bg=bg)
        for ci,k in enumerate(["P","R","F1"],2):
            cell=_dat(ws,r,ci,res[k],bg=bg,h="center",fmt="0.0%")
            if res[k]==1.0: cell.font=_font(bold=True,color=C_GREEN)
        r+=1
    _dat(ws,r,1,f"Gold set: {rule['TP']+rule['FN']+rule['FP']+rule['TN']} dokumen terverifikasi manual",
         bg=BI_BLUE_SOFT,italic=True)
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=4); r+=2

    # C. Dekomposisi
    _title(ws,r,"C.  DEKOMPOSISI SEKTORAL (GOLD GSS BOND)",4); r+=1
    _hdr(ws,r,1,"Status",bg=C_DARK,h="left"); _hdr(ws,r,2,"Jumlah",bg=C_DARK)
    _hdr(ws,r,3,"Persentase",bg=C_DARK); _hdr(ws,r,4,"Penjelasan",bg=C_DARK,h="left"); r+=1
    for label,cnt,bg,ket in [
        ("Terdekomposisi (sektor UoP terbaca)",bcnt["terdekomposisi"],C_GREEN_S,"Sektor didanai teridentifikasi dari teks prospektus"),
        ("Tanpa sektor (UoP terlalu tipis)",bcnt["sektor_tak_terverifikasi"],C_ORG_S,"Kelas dari nama instrumen; UoP tidak cukup detail"),
        ("Level-0 struktural (SL / Wakaf)",bcnt["level0"],C_SUS_S,"Sustainability Linked / Sukuk Wakaf"),
        ("TOTAL gold set",n_s,BI_BLUE_SOFT,""),
    ]:
        _dat(ws,r,1,label,bg=bg,bold=(label=="TOTAL gold set"))
        _dat(ws,r,2,cnt,bg=bg,h="center",bold=(label=="TOTAL gold set"))
        _dat(ws,r,3,cnt/n_s if n_s else 0,bg=bg,h="center",fmt="0%",bold=(label=="TOTAL gold set"))
        _dat(ws,r,4,ket,bg=bg,wrap=True); r+=1
    r+=1

    # D. Verifikasi
    _title(ws,r,"D.  STATUS VERIFIKASI KLAIM",4); r+=1
    _hdr(ws,r,1,"Status Verifikasi",bg=C_DARK,h="left"); _hdr(ws,r,2,"Jumlah",bg=C_DARK)
    _hdr(ws,r,3,"Share",bg=C_DARK); _hdr(ws,r,4,"Interpretasi",bg=C_DARK,h="left"); r+=1
    for label,cnt,bg,interp in [
        ("Terverifikasi",6,C_GREEN_S,"Bukti UoP konsisten dengan kelas instrumen"),
        ("Sebagian",12,C_SOC_S,"Sustainability: hanya satu dimensi terbukti (umumnya sosial)"),
        ("Tidak terverifikasi (UoP tipis)",6,C_ORG_S,"Kelas dari nama; UoP tidak beri bukti sektoral"),
        ("Level-0 (struktural)",3,C_SUS_S,"SL/Wakaf — metode verifikasi berbeda"),
    ]:
        _dat(ws,r,1,label,bg=bg,bold=True); _dat(ws,r,2,cnt,bg=bg,h="center",bold=True)
        _dat(ws,r,3,cnt/n_s if n_s else 0,bg=bg,h="center",fmt="0%")
        _dat(ws,r,4,interp,bg=bg,wrap=True); ws.row_dimensions[r].height=24; r+=1

    # Link ke sheet terkait
    r+=1
    _dat(ws,r,1,"Lihat detail di:",bg=BI_BLUE_SOFT,bold=True); r+=1
    for sh,label in [
        ("2. Sensus Pasar GSS","→ Sensus Pasar GSS"),
        ("4. Dekomposisi Sektoral","→ Dekomposisi Sektoral"),
        ("5. Evaluasi Klasifikasi","→ Evaluasi Klasifikasi"),
        ("9. Pivot Interaktif","→ Pivot Table Interaktif"),
    ]:
        _link(ws,r,1,label,sh,bg=BI_BLUE_SOFT,size=10); r+=1

    _no_gridlines(ws)
    print("  sheet 1 — Ringkasan Eksekutif")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 2 — Sensus Pasar GSS  (TblSensus)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_sensus(wb,census):
    ws=wb.create_sheet("2. Sensus Pasar GSS")
    _cw(ws,1,24); _cw(ws,2,16); _cw(ws,3,22); _cw(ws,4,20); _cw(ws,5,14); _cw(ws,6,42)

    r=1
    _back(ws,r,1,6); r+=1
    _title(ws,r,"SENSUS PASAR EBUS GSS KORPORASI — Berdasarkan Nama & Kode BEI",6); r+=1

    # Excel Table header
    HDR=["Kelas GSS","Jumlah Instrumen","Outstanding (Rp T)","Share Universe (%)","Kode BEI","Keterangan"]
    tbl_start=r
    for ci,h in enumerate(HDR,1): _hdr(ws,r,ci,h,bg=C_DARK,size=9)
    r+=1

    KET={
        "Green":"GN / GNCN — Berwawasan Lingkungan",
        "Social":"SOCN / SECN — Berwawasan Sosial",
        "Sustainability":"SLCN (UoP) — Campuran Lingkungan & Sosial",
        "Sustainability Linked":"SL / SLCN — Berbasis KPI/target kinerja",
        "TOTAL":"",
    }
    BEI_CODE={"Green":"GN, GNCN","Social":"SOCN, SECN","Sustainability":"SLCN (UoP)",
              "Sustainability Linked":"SL, SLCN","TOTAL":"—"}

    data_rows=[]
    for row in census:
        gt=row["gss_type"]
        is_total=gt=="TOTAL"
        bg=BI_BLUE_SOFT if is_total else CLASS_SOFT.get(gt,C_GRAY_S)
        _dat(ws,r,1,gt,bg=bg,bold=is_total,color=CLASS_HEADER.get(gt,BI_BLUE) if not is_total else BI_BLUE)
        _dat(ws,r,2,int(row["n_instrumen"]),bg=bg,h="center",bold=is_total)
        _dat(ws,r,3,float(row["total_outstanding_triliun"]),bg=bg,h="right",fmt="#,##0.00",bold=is_total)
        _dat(ws,r,4,float(row["share_universe_pct"])/100,bg=bg,h="center",fmt="0.00%",bold=is_total)
        _dat(ws,r,5,BEI_CODE.get(gt,""),bg=bg,h="center")
        _dat(ws,r,6,KET.get(gt,""),bg=bg,wrap=True)
        data_rows.append(r); r+=1

    tbl_end=data_rows[-1]
    _add_table(ws,"TblSensus",f"A{tbl_start}:{get_column_letter(6)}{tbl_end}","TableStyleMedium2")
    ws.freeze_panes="A3"; _no_gridlines(ws)
    print("  sheet 2 — Sensus Pasar GSS")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 3 — Daftar 82 GSS Berlabel  (TblGSS)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_daftar(wb):
    ws=wb.create_sheet("3. Daftar 82 GSS Berlabel")
    rows=_load("idx_gss_all_20260618_140427.csv")
    _cw(ws,1,5); _cw(ws,2,14); _cw(ws,3,8); _cw(ws,4,58)
    _cw(ws,5,18); _cw(ws,6,9); _cw(ws,7,14); _cw(ws,8,10); _cw(ws,9,18)

    r=1
    _back(ws,r,1,9); r+=1
    _title(ws,r,f"DAFTAR {len(rows)} EBUS GSS BERLABEL — Klik emiten untuk lihat dekomposisi sektoral",9); r+=1

    HDR=["No","Kode Bond","Emiten","Nama Instrumen","Kelas GSS",
         "Kode BEI","Jatuh Tempo","Rating","Outstanding (Rp M)"]
    tbl_start=r
    for ci,h in enumerate(HDR,1): _hdr(ws,r,ci,h,bg=C_DARK,size=9,h="center")
    r+=1

    for i,row in enumerate(rows,1):
        gt=row.get("gss_type","")
        bg=CLASS_SOFT.get(gt,C_GRAY_S) if i%2==1 else "FFFFFF"
        mat=row.get("MatureDate","")[:10] if row.get("MatureDate") else ""
        try: os_m=round(float(row.get("Outstanding") or 0)/1e6,1)
        except: os_m=""
        iss=row.get("IssuerCode","")
        _dat(ws,r,1,i,bg=bg,h="center")
        _dat(ws,r,2,row.get("BondId",""),bg=bg)
        # Hyperlink emiten → sheet dekomposisi
        cell=ws.cell(row=r,column=3,value=iss)
        cell.hyperlink=f"#'4. Dekomposisi Sektoral'!A1"
        cell.font=_font(color="0563C1",underline="single",bold=True)
        cell.alignment=_align(h="center"); cell.border=_thin()
        if bg!="FFFFFF": cell.fill=_fill(bg)
        _dat(ws,r,4,row.get("BondName",""),bg=bg,wrap=True)
        _dat(ws,r,5,gt,bg=CLASS_SOFT.get(gt,bg),h="center",bold=True,color=CLASS_HEADER.get(gt,"000000"))
        _dat(ws,r,6,row.get("bei_code",""),bg=bg,h="center")
        _dat(ws,r,7,mat,bg=bg,h="center")
        _dat(ws,r,8,row.get("Rating",""),bg=bg,h="center")
        _dat(ws,r,9,os_m,bg=bg,h="right",fmt="#,##0.0"); r+=1

    _add_table(ws,"TblGSS",f"A{tbl_start}:{get_column_letter(9)}{r-1}","TableStyleMedium2")
    ws.freeze_panes="A4"; _no_gridlines(ws)
    print("  sheet 3 — Daftar 82 GSS Berlabel")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 4 — Dekomposisi Sektoral  (TblDekomposisi)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_dekomposisi(wb,sectors):
    ws=wb.create_sheet("4. Dekomposisi Sektoral")
    _cw(ws,1,5); _cw(ws,2,9); _cw(ws,3,22); _cw(ws,4,16); _cw(ws,5,20)
    _cw(ws,6,42); _cw(ws,7,24); _cw(ws,8,10)

    r=1
    _back(ws,r,1,8); r+=1
    _title(ws,r,f"DEKOMPOSISI SEKTORAL USE-OF-PROCEEDS — {len(sectors)} Gold GSS Bond",8); r+=1
    _dat(ws,r,1,"Filter: gunakan tombol ▼ di header tabel untuk memfilter per Kelas, Status, atau Emiten",
         bg=BI_BLUE_SOFT,italic=True)
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=8); r+=1

    HDR=["No","Emiten","Berkas Prospektus","Kelas GSS","Status Dekomposisi",
         "Kategori Sektoral Teridentifikasi","Status Verifikasi","Keyakinan"]
    tbl_start=r
    for ci,h in enumerate(HDR,1): _hdr(ws,r,ci,h,bg=C_DARK,size=9)
    r+=1

    BKT_BG={"terdekomposisi":C_GREEN_S,"sektor_tak_terverifikasi":C_ORG_S,"level0":C_SUS_S}
    BKT_LBL={"terdekomposisi":"Terdekomposisi","sektor_tak_terverifikasi":"Tanpa Sektor","level0":"Level-0 Struktural"}
    VS_BG={"Terverifikasi":C_GREEN_S,"Sebagian":C_SOC_S,"Level-0 (struktural)":C_SUS_S}

    for i,row in enumerate(sectors,1):
        bkt=row.get("bucket",""); gc=row.get("gss_class","")
        bg=BKT_BG.get(bkt,"FFFFFF") if i%2==1 else "FFFFFF"
        skeys=row.get("sector_keys","")
        sektor="\n".join(f"• {CAT_LABEL.get(k,k)}" for k in skeys.split("|") if k) if skeys else "—"
        vs=row.get("verification_status","")
        try: conf=float(row.get("confidence","0") or 0)
        except: conf=0.0

        _dat(ws,r,1,i,bg=bg,h="center")
        _dat(ws,r,2,row.get("issuer",""),bg=bg,h="center",bold=True)
        _dat(ws,r,3,row.get("pdf",""),bg=bg,wrap=True)
        gc_disp={"green":"Green","social":"Social","sustainability":"Sustainability","sustainability_linked":"Sust. Linked"}.get(gc,gc)
        _dat(ws,r,4,gc_disp,bg=CLASS_SOFT.get(gc,bg),h="center",bold=True,color=CLASS_HEADER.get(gc,"000000"))
        _dat(ws,r,5,BKT_LBL.get(bkt,bkt),bg=bg,h="center")
        _dat(ws,r,6,sektor,bg=bg,wrap=True)
        _dat(ws,r,7,vs,bg=VS_BG.get(vs,bg),h="center",bold=True)
        _dat(ws,r,8,conf,bg=bg,h="center",fmt="0%")
        n_k=len(skeys.split("|")) if skeys else 0
        ws.row_dimensions[r].height=max(18,16*n_k); r+=1

    _add_table(ws,"TblDekomposisi",f"A{tbl_start}:{get_column_letter(8)}{r-1}","TableStyleMedium9")
    ws.freeze_panes="A5"; _no_gridlines(ws)
    print("  sheet 4 — Dekomposisi Sektoral")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 5 — Evaluasi Klasifikasi
# ══════════════════════════════════════════════════════════════════════════════
def sheet_evaluasi(wb,comparison):
    ws=wb.create_sheet("5. Evaluasi Klasifikasi")
    _cw(ws,1,34); _cw(ws,2,14); _cw(ws,3,14); _cw(ws,4,14)
    _cw(ws,5,14); _cw(ws,6,14); _cw(ws,7,14)

    rule=_prf(comparison,"pred_rule"); ml=_prf(comparison,"pred_ml")
    n=len(comparison)

    r=1
    _back(ws,r,1,7); r+=1
    _title(ws,r,"EVALUASI KINERJA KLASIFIKASI GSS — Rule-based vs ML Semantik",7,size=12)
    ws.row_dimensions[r].height=24; r+=2

    # P/R/F1
    _title(ws,r,"A.  METRIK KINERJA UTAMA",7); r+=1
    _hdr(ws,r,1,"Metode",bg=C_DARK,h="left")
    for ci,h in enumerate(["Precision","Recall","F1-Score","TP","FP (Salah GSS)","FN (Terlewat)"],2):
        _hdr(ws,r,ci,h,bg=C_DARK)
    r+=1
    for met,res,bg in [("Rule-based (baseline kata kunci POJK)",rule,C_GRAY_S),
                        ("ML Semantik — hibrida leksikal+semantik (final)",ml,C_GREEN_S)]:
        _dat(ws,r,1,met,bg=bg)
        for ci,k in enumerate(["P","R","F1"],2):
            c=_dat(ws,r,ci,res[k],bg=bg,h="center",fmt="0.0%")
            if res[k]==1.0: c.font=_font(bold=True,color=C_GREEN)
        _dat(ws,r,5,res["TP"],bg=bg,h="center")
        _dat(ws,r,6,res["FP"],bg=bg,h="center",color=C_ORANGE if res["FP"]>0 else "000000")
        _dat(ws,r,7,res["FN"],bg=bg,h="center",color=C_ORANGE if res["FN"]>0 else "000000")
        r+=1
    r+=1

    # Confusion matrix — dua berdampingan
    for label,res in [("B.  CONFUSION MATRIX — Rule-based",rule),("C.  CONFUSION MATRIX — ML Semantik",ml)]:
        _title(ws,r,label,7); r+=1
        _hdr(ws,r,1,"Prediksi ↓ / Aktual →",bg=C_DARK,h="left",span=2)
        _hdr(ws,r,3,"Aktual: GSS",bg=C_DARK); _hdr(ws,r,4,"Aktual: Non-GSS",bg=C_DARK)
        _hdr(ws,r,5,"Total",bg=C_DARK); _hdr(ws,r,6,"Keterangan",bg=C_DARK,span=2); r+=1
        cm=[
            ("Prediksi: GSS (Positif)",res["TP"],res["FP"],res["TP"]+res["FP"],
             f"TP={res['TP']} Benar | FP={res['FP']} Salah identifikasi",C_GREEN_S),
            ("Prediksi: Non-GSS (Negatif)",res["FN"],res["TN"],res["FN"]+res["TN"],
             f"FN={res['FN']} Terlewat | TN={res['TN']} Benar non-GSS",C_ORG_S if res["FN"]>0 else C_GRAY_S),
            ("Total Aktual",res["TP"]+res["FN"],res["FP"]+res["TN"],n,"",BI_BLUE_SOFT),
        ]
        for lbl,v1,v2,tot,ket,bg in cm:
            bold=(lbl=="Total Aktual")
            _dat(ws,r,1,lbl,bg=bg,bold=bold); ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=2)
            _dat(ws,r,3,v1,bg=bg,h="center",bold=bold,color=C_GREEN if (v1>0 and lbl!="Total Aktual") else "000000")
            _dat(ws,r,4,v2,bg=bg,h="center",bold=bold,color=C_ORANGE if (v2>0 and "Positif" in lbl) else "000000")
            _dat(ws,r,5,tot,bg=bg,h="center",bold=bold)
            _dat(ws,r,6,ket,bg=bg,wrap=True); ws.merge_cells(start_row=r,start_column=6,end_row=r,end_column=7)
            ws.row_dimensions[r].height=22; r+=1
        r+=1

    # Interpretasi
    _title(ws,r,"D.  INTERPRETASI & CATATAN KEJUJURAN",7); r+=1
    notes=[
        f"Precision = dari semua diprediksi GSS, berapa yang benar? Rule: {rule['P']:.1%} → banyak FP. ML: {ml['P']:.1%} = sempurna.",
        f"Recall = dari semua GSS nyata, berapa yang ditemukan? Rule: {rule['R']:.1%} (masih tinggi). ML: {ml['R']:.1%} = sempurna.",
        "False positive utama rule-based: frasa 'Berkelanjutan' dalam nama Penawaran Umum Berkelanjutan (PUB) — administratif, bukan GSS.",
        "Akurasi sempurna (F1=1.0) di gold set 93 dokumen tidak menjamin generalisasi penuh ke instrumen yang belum pernah dilihat.",
        "Gold set masih timpang antar-kelas (Green & SL sedikit) → keandalan lebih tinggi untuk Sustainability & Social.",
    ]
    for note in notes:
        _dat(ws,r,1,f"• {note}",bg=BI_BLUE_SOFT,wrap=True)
        ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=7)
        ws.row_dimensions[r].height=32; r+=1

    _no_gridlines(ws)
    print("  sheet 5 — Evaluasi Klasifikasi")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 6 — Detail Gold Set  (TblGoldSet)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_gold(wb,comparison):
    ws=wb.create_sheet("6. Detail Gold Set")
    _cw(ws,1,5); _cw(ws,2,8); _cw(ws,3,11); _cw(ws,4,11); _cw(ws,5,11)
    _cw(ws,6,14); _cw(ws,7,12); _cw(ws,8,12); _cw(ws,9,11); _cw(ws,10,42)

    r=1
    _back(ws,r,1,10); r+=1
    _title(ws,r,f"DETAIL GOLD SET — {len(comparison)} Dokumen Prospektus Terverifikasi",10); r+=1
    _dat(ws,r,1,"Hijau = keduanya benar · Kuning = rule salah, ML benar · Oranye = ML salah",
         bg=BI_BLUE_SOFT,italic=True)
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=10); r+=1

    HDR=["No","Emiten","Aktual (Gold)","Pred. Rule","Pred. ML",
         "Kelas ML","Ada Framing","Deteksi Judul","Keyakinan ML","Berkas Prospektus"]
    tbl_start=r
    for ci,h in enumerate(HDR,1): _hdr(ws,r,ci,h,bg=C_DARK,size=9,h="center",wrap=True)
    ws.row_dimensions[r].height=28; r+=1

    for i,row in enumerate(comparison,1):
        gold=row.get("gold",""); pr=row.get("pred_rule",""); pm=row.get("pred_ml","")
        mc=row.get("ml_class","")
        if pr==gold and pm==gold: bg=C_GREEN_S if i%2==1 else "F0F8F0"
        elif pm==gold: bg=C_SOC_S
        else: bg=C_ORG_S
        try: conf=float(row.get("ml_conf","0") or 0)
        except: conf=0.0
        framing=bool(row.get("ml_framing_body","").strip())
        title_gss=row.get("ml_title_gss","")=="True"
        _dat(ws,r,1,i,bg=bg,h="center")
        _dat(ws,r,2,row.get("issuer",""),bg=bg,h="center",bold=True)
        _dat(ws,r,3,gold,bg=bg,h="center",color=C_GREEN if gold=="GSS" else C_GRAY)
        _dat(ws,r,4,pr,bg=bg,h="center",color=C_GREEN if pr==gold else C_ORANGE,bold=(pr!=gold))
        _dat(ws,r,5,pm,bg=bg,h="center",color=C_GREEN if pm==gold else C_ORANGE,bold=(pm!=gold))
        _dat(ws,r,6,mc,bg=CLASS_SOFT.get(mc,bg),h="center")
        _dat(ws,r,7,"Ya" if framing else "Tidak",bg=bg,h="center",color=C_GREEN if framing else C_GRAY)
        _dat(ws,r,8,"Ya" if title_gss else "Tidak",bg=bg,h="center",color=C_GREEN if title_gss else C_GRAY)
        _dat(ws,r,9,conf,bg=bg,h="center",fmt="0%")
        _dat(ws,r,10,row.get("pdf",""),bg=bg,wrap=True)
        ws.row_dimensions[r].height=16; r+=1

    _add_table(ws,"TblGoldSet",f"A{tbl_start}:{get_column_letter(10)}{r-1}","TableStyleMedium2")
    ws.freeze_panes="A5"; _no_gridlines(ws)
    print("  sheet 6 — Detail Gold Set")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 7 — Taksonomi 15 Kategori  (TblTaksonomi)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_taksonomi(wb):
    ws=wb.create_sheet("7. Taksonomi 15 Kategori")
    _cw(ws,1,5); _cw(ws,2,16); _cw(ws,3,28); _cw(ws,4,14); _cw(ws,5,50); _cw(ws,6,40)

    r=1
    _back(ws,r,1,6); r+=1
    _title(ws,r,"TAKSONOMI 15 KATEGORI ELIGIBLE GSS — Dasar Klasifikasi",6); r+=1
    _dat(ws,r,1,"Berdasarkan: POJK 18/2023 | ICMA Green & Social Bond Principles | SDG Government Securities Framework (DJPPR)",
         bg=BI_BLUE_SOFT,italic=True,bold=True)
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=6); r+=1

    HDR=["No","Kode Kategori","Nama Kategori","Kelompok","Kata Kunci Utama (contoh)","Contoh Instrumen / Proyek"]
    tbl_start=r
    for ci,h in enumerate(HDR,1): _hdr(ws,r,ci,h,bg=C_DARK,size=9,h="center",wrap=True)
    ws.row_dimensions[r].height=24; r+=1

    TAX=[
        ("renewable_energy","Energi Terbarukan","Lingkungan",
         "energi terbarukan, panel surya, PLTS, tenaga angin, panas bumi, PLTA, biomassa, biogas",
         "PLTS atap, PLTA run-of-river, PLTB, geothermal power plant"),
        ("energy_efficiency","Efisiensi Energi","Lingkungan",
         "efisiensi energi, konservasi energi, smart grid, retrofit, kogenerasi",
         "Retrofit LED gedung, modernisasi trafo distribusi, audit energi"),
        ("green_building","Bangunan Hijau","Lingkungan",
         "bangunan hijau, green building, greenship, EDGE certification, LEED",
         "Gedung perkantoran bersertifikat Greenship/EDGE"),
        ("green_tourism","Pariwisata Berkelanjutan","Lingkungan",
         "pariwisata berkelanjutan, ekowisata, sustainable tourism",
         "Resort eco-certified, destinasi wisata berbasis konservasi"),
        ("sustainable_transport","Transportasi Berkelanjutan","Lingkungan",
         "MRT, LRT, KRL, BRT, kendaraan listrik, electric vehicle, transportasi massal",
         "Armada bus listrik, perluasan jaringan MRT/LRT"),
        ("waste_management","Pengelolaan Sampah & Limbah","Lingkungan",
         "pengelolaan sampah, waste to energy, PLTSA, daur ulang, 3R",
         "Fasilitas sampah jadi energi, IPAL industri"),
        ("water_management","Pengelolaan Air & Air Limbah","Lingkungan",
         "SPAM, IPAL, air bersih, pengelolaan air limbah, sanitasi, irigasi",
         "Pembangunan SPAM regional, rehabilitasi IPAL kota"),
        ("natural_resources","Pengelolaan SDA Berkelanjutan","Lingkungan (Blue)",
         "kehutanan lestari, reboisasi, konservasi, perikanan berkelanjutan, akuakultur",
         "Rehabilitasi mangrove, sertifikasi hutan FSC, budidaya ikan"),
        ("climate_resilience","Ketahanan Iklim & PRB","Lingkungan",
         "ketahanan iklim, adaptasi perubahan iklim, pengurangan risiko bencana, pengendalian banjir",
         "Infrastruktur tahan banjir, sistem peringatan dini bencana"),
        ("basic_infrastructure","Infrastruktur Dasar Terjangkau","Sosial",
         "infrastruktur dasar, akses air minum, sanitasi dasar, listrik perdesaan, elektrifikasi",
         "Elektrifikasi daerah terpencil, jaringan air minum perdesaan"),
        ("essential_services","Akses Layanan Esensial","Sosial",
         "akses kesehatan, rumah sakit, puskesmas, pendidikan, beasiswa",
         "Puskesmas di daerah 3T, bantuan beasiswa pendidikan"),
        ("affordable_housing","Perumahan Terjangkau","Sosial",
         "perumahan terjangkau, affordable housing, rumah subsidi, MBR, KPR FLPP",
         "KPR bersubsidi MBR, pembangunan rusunami"),
        ("employment_msme","Penciptaan Lapangan Kerja & UMKM","Sosial",
         "UMKM, usaha mikro, Kredit Usaha Rakyat, KUR, microfinance",
         "Program KUR UMKM, pembiayaan ultra-mikro"),
        ("food_security","Ketahanan Pangan","Sosial",
         "ketahanan pangan, food security, produktivitas pertanian",
         "Gudang pangan, rantai pasok pertanian, alat mesin pertanian"),
        ("socioeconomic","Pemberdayaan Sosial-Ekonomi","Sosial",
         "pemberdayaan, pengentasan kemiskinan, inklusi keuangan, financial inclusion",
         "Program pemberdayaan perempuan, inklusi keuangan daerah terpencil"),
    ]
    for i,(key,nama,kel,kw,contoh) in enumerate(TAX,1):
        bg=C_GREEN_S if "Lingkungan" in kel else C_SOC_S
        bg=bg if i%2==1 else ("F0F8F0" if "Lingkungan" in kel else "FFFDE7")
        _dat(ws,r,1,i,bg=bg,h="center",bold=True)
        _dat(ws,r,2,key,bg=bg)
        _dat(ws,r,3,nama,bg=bg,bold=True)
        _dat(ws,r,4,kel,bg=bg,h="center")
        _dat(ws,r,5,kw,bg=bg,wrap=True)
        _dat(ws,r,6,contoh,bg=bg,wrap=True)
        ws.row_dimensions[r].height=36; r+=1

    _add_table(ws,"TblTaksonomi",f"A{tbl_start}:{get_column_letter(6)}{r-1}","TableStyleMedium9")
    r+=1

    # Level-0
    _title(ws,r,"LEVEL-0 — SINYAL STRUKTUR (Diperiksa Sebelum Kategori UoP)",6); r+=1
    _hdr(ws,r,1,"Tipe",bg=C_DARK,span=2)
    _hdr(ws,r,3,"Kata Kunci Pemicu",bg=C_DARK,span=2)
    _hdr(ws,r,5,"Implikasi Klasifikasi",bg=C_DARK,span=2); r+=1
    for tipe,kw,impl,bg in [
        ("Sustainability-Linked","sustainability-linked, terkait keberlanjutan, KPI, SPT, step-up, coupon ratchet",
         "Diklasifikasi Sustainability Linked — bukan berdasar UoP",C_SL_S),
        ("Sukuk Wakaf","wakaf, waqf, CWLS, nazhir, ikrar wakaf, mauquf",
         "Diklasifikasi Wakaf — pelaporan berbeda dari GSS reguler",C_SUS_S),
    ]:
        _dat(ws,r,1,tipe,bg=bg,bold=True,h="center"); ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=2)
        _dat(ws,r,3,kw,bg=bg,wrap=True); ws.merge_cells(start_row=r,start_column=3,end_row=r,end_column=4)
        _dat(ws,r,5,impl,bg=bg,wrap=True); ws.merge_cells(start_row=r,start_column=5,end_row=r,end_column=6)
        ws.row_dimensions[r].height=36; r+=1

    ws.freeze_panes="A5"; _no_gridlines(ws)
    print("  sheet 7 — Taksonomi 15 Kategori")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 8 — SBN GSS Sovereign  (TblSBN)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_sbn(wb):
    ws=wb.create_sheet("8. SBN GSS Sovereign (Acuan)")
    rows=_load("sbn_gss_lookup.csv")
    _cw(ws,1,5); _cw(ws,2,28); _cw(ws,3,22); _cw(ws,4,14); _cw(ws,5,14)
    _cw(ws,6,14); _cw(ws,7,14); _cw(ws,8,10); _cw(ws,9,16); _cw(ws,10,12); _cw(ws,11,40)

    r=1
    _back(ws,r,1,11); r+=1
    _title(ws,r,f"ACUAN SBN GSS SOVEREIGN — {len(rows)} Instrumen (Taksonomi Ground Truth)",11); r+=1
    _dat(ws,r,1,"SBN GSS sudah diaudit BPK & tercatat di SRN-PPI → tidak perlu ML. Digunakan sebagai taksonomi acuan & ground truth AN ini.",
         bg=BI_BLUE_SOFT,italic=True)
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=11); r+=1

    HDR=["No","Seri / Nama","Jenis Instrumen","Pasar","Kategori GSS",
         "Tema","Tanggal Emisi","Mata Uang","Nominal","Tenor","Framework / Sumber"]
    tbl_start=r
    for ci,h in enumerate(HDR,1): _hdr(ws,r,ci,h,bg=C_DARK,size=9,h="center",wrap=True)
    ws.row_dimensions[r].height=28; r+=1

    KAT_BG={"Green":C_GREEN_S,"Sustainability":C_SUS_S,"Social":C_SOC_S}
    for i,row in enumerate(rows,1):
        kat=row.get("kategori_gss",""); bg=KAT_BG.get(kat,C_GRAY_S) if i%2==1 else "FFFFFF"
        fw=row.get("framework_induk",""); src=row.get("sumber",""); cat=row.get("catatan","")
        combo=f"{fw} | {src}"+( f" | {cat}" if cat else "")
        _dat(ws,r,1,int(row.get("no",i)),bg=bg,h="center")
        _dat(ws,r,2,row.get("seri",""),bg=bg,bold=True,wrap=True)
        _dat(ws,r,3,row.get("jenis_instrumen",""),bg=bg)
        _dat(ws,r,4,row.get("pasar",""),bg=bg,h="center")
        _dat(ws,r,5,kat,bg=KAT_BG.get(kat,bg),h="center",bold=True,
             color=CLASS_HEADER.get(kat,"000000"))
        _dat(ws,r,6,row.get("tema",""),bg=bg,h="center")
        _dat(ws,r,7,row.get("tanggal_emisi",""),bg=bg,h="center")
        _dat(ws,r,8,row.get("mata_uang",""),bg=bg,h="center")
        _dat(ws,r,9,row.get("nominal",""),bg=bg,h="right")
        _dat(ws,r,10,row.get("tenor",""),bg=bg,h="center")
        _dat(ws,r,11,combo,bg=bg,wrap=True)
        ws.row_dimensions[r].height=20; r+=1

    _add_table(ws,"TblSBN",f"A{tbl_start}:{get_column_letter(11)}{r-1}","TableStyleMedium2")
    ws.freeze_panes="A5"; _no_gridlines(ws)
    print("  sheet 8 — SBN GSS Sovereign")


# ══════════════════════════════════════════════════════════════════════════════
# Sheet tersembunyi — Data kategori diperluas (sumber pivot kategori × emiten)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_kat_expanded(wb,sectors):
    ws=wb.create_sheet("_Data_Kategori")
    ws.sheet_state="hidden"
    HDR=["Emiten","Kelas GSS","Status Dekomposisi","Kategori","Kelompok","Status Verifikasi","Keyakinan"]
    for ci,h in enumerate(HDR,1):
        ws.cell(row=1,column=ci,value=h).font=_font(bold=True)

    r=2
    for s in sectors:
        skeys=s.get("sector_keys","")
        gc=s.get("gss_class","")
        gc_disp={"green":"Green","social":"Social","sustainability":"Sustainability","sustainability_linked":"Sust. Linked"}.get(gc,gc)
        bkt={"terdekomposisi":"Terdekomposisi","sektor_tak_terverifikasi":"Tanpa Sektor","level0":"Level-0"}.get(s.get("bucket",""),s.get("bucket",""))
        if skeys:
            for k in skeys.split("|"):
                if k:
                    try: conf=float(s.get("confidence","0") or 0)
                    except: conf=0.0
                    vals=[s.get("issuer",""),gc_disp,bkt,CAT_LABEL.get(k,k),CAT_BUCKET.get(k,""),s.get("verification_status",""),conf]
                    for ci,v in enumerate(vals,1): ws.cell(row=r,column=ci,value=v)
                    r+=1
        else:
            try: conf=float(s.get("confidence","0") or 0)
            except: conf=0.0
            vals=[s.get("issuer",""),gc_disp,bkt,"—","—",s.get("verification_status",""),conf]
            for ci,v in enumerate(vals,1): ws.cell(row=r,column=ci,value=v)
            r+=1

    if r>2:
        _add_table(ws,"TblKatExpanded",f"A1:{get_column_letter(7)}{r-1}","TableStyleLight1")
    print(f"  sheet _Data_Kategori — {r-2} baris diperluas")
    return r-1  # last data row


# ══════════════════════════════════════════════════════════════════════════════
# Sheet 9 — Placeholder pivot (diisi oleh win32com setelah save)
# ══════════════════════════════════════════════════════════════════════════════
def sheet_pivot_placeholder(wb):
    ws=wb.create_sheet("9. Pivot Interaktif")
    _cw(ws,1,30); _cw(ws,2,24); _cw(ws,3,20); _cw(ws,4,20); _cw(ws,5,20); _cw(ws,6,20)
    r=1
    _back(ws,r,1,6); r+=1
    _title(ws,r,"PIVOT TABLE INTERAKTIF — 3 Analisis Silang Siap Pakai",6,size=12)
    ws.row_dimensions[r].height=24; r+=1

    # Panduan singkat — hanya baris 3 saja, baris 4+ dikosongkan untuk pivot
    tips_row=[
        "Pivot 1 (mulai A8):  Status Verifikasi x Kelas GSS  |  Sumber: sheet 4 Dekomposisi Sektoral",
        "Pivot 2 (mulai A26): Kategori Sektoral × Emiten     |  Sumber: sheet _Data_Kategori (tersembunyi)",
        "Pivot 3 (mulai A50): Outstanding × Emiten × Kelas   |  Sumber: sheet 3 Daftar 82 GSS Berlabel",
        "Tips: klik kanan di pivot → Refresh | Double-klik nilai untuk drill-down ke data detail",
    ]
    for tip in tips_row:
        _dat(ws,r,1,tip,bg=BI_BLUE_SOFT,italic=True)
        ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=6)
        ws.row_dimensions[r].height=18; r+=1

    # Baris 5+ dibiarkan kosong — COM akan insert pivot tables di sini
    _no_gridlines(ws)
    print("  sheet 9 — Pivot Interaktif (placeholder)")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Tambahkan pivot table nyata via win32com
# ══════════════════════════════════════════════════════════════════════════════
def add_pivot_tables(path: str):
    try:
        import win32com.client as win32
        import pywintypes
    except ImportError:
        print("  ! win32com tidak tersedia — lewati pivot table")
        return

    print("\n[2/2] Menambahkan Pivot Table via Excel COM...")
    xl=None
    try:
        xl=win32.Dispatch("Excel.Application")
        xl.Visible=False
        xl.DisplayAlerts=False
        abs_path=os.path.abspath(path)
        wb=xl.Workbooks.Open(abs_path)

        # Konstanta Excel
        xlDatabase=1; xlRowField=1; xlColumnField=2; xlDataField=4
        xlCount=-4112; xlSum=-4157

        pivot_ws=wb.Sheets("9. Pivot Interaktif")

        xlUp=-4162  # xlDirection
        from openpyxl.utils import get_column_letter

        def _src(sheet_name, hdr_row, ncols):
            """Hitung range sumber data: dari baris header sampai baris data terakhir."""
            ws=wb.Sheets(sheet_name)
            last=ws.Cells(ws.Rows.Count, 1).End(xlUp).Row
            last=max(last, hdr_row+1)  # minimal 1 data row
            col_last=get_column_letter(ncols)
            sn=sheet_name.replace("'","''")
            return f"'{sn}'!$A${hdr_row}:${col_last}${last}"

        def _make_pivot(pc, dest_cell, tbl_name, style="PivotStyleMedium2"):
            return pc.CreatePivotTable(TableDestination=pivot_ws.Range(dest_cell),
                                       TableName=tbl_name)

        # ── Pivot 1: Status Verifikasi × Kelas GSS ──────────────────────────
        # Sheet "4. Dekomposisi Sektoral": header=row 4, ncols=8
        print("    Pivot 1: Status Verifikasi per Kelas GSS...")
        try:
            src1=_src("4. Dekomposisi Sektoral", 4, 8)
            pc1=wb.PivotCaches().Create(SourceType=xlDatabase, SourceData=src1)
            pt1=_make_pivot(pc1,"A8","PivotVerifikasi")
            pt1.PivotFields("Kelas GSS").Orientation=xlRowField
            pt1.PivotFields("Status Verifikasi").Orientation=xlColumnField
            pt1.AddDataField(pt1.PivotFields("Emiten"),"Jumlah Bond",xlCount)
            pt1.ColumnGrand=True; pt1.RowGrand=True
            pt1.TableStyle2="PivotStyleMedium9"
            print("    OK Pivot 1 berhasil")
        except Exception as e:
            print(f"    ! Pivot 1 gagal: {e}")

        # ── Pivot 2: Kategori × Emiten ──────────────────────────────────────
        # Sheet "_Data_Kategori": header=row 1, ncols=7
        print("    Pivot 2: Kategori Sektoral per Emiten...")
        try:
            src2=_src("_Data_Kategori", 1, 7)
            pc2=wb.PivotCaches().Create(SourceType=xlDatabase, SourceData=src2)
            pt2=_make_pivot(pc2,"A26","PivotKategori")
            pt2.PivotFields("Emiten").Orientation=xlRowField
            pt2.PivotFields("Kelompok").Orientation=xlColumnField
            pt2.AddDataField(pt2.PivotFields("Kategori"),"Jumlah Kategori",xlCount)
            pt2.ColumnGrand=True; pt2.RowGrand=True
            pt2.TableStyle2="PivotStyleMedium2"
            print("    OK Pivot 2 berhasil")
        except Exception as e:
            print(f"    ! Pivot 2 gagal: {e}")

        # ── Pivot 3: Outstanding per Emiten per Kelas ────────────────────────
        # Sheet "3. Daftar 82 GSS Berlabel": header=row 3, ncols=9
        print("    Pivot 3: Outstanding per Emiten per Kelas...")
        try:
            src3=_src("3. Daftar 82 GSS Berlabel", 3, 9)
            pc3=wb.PivotCaches().Create(SourceType=xlDatabase, SourceData=src3)
            pt3=_make_pivot(pc3,"A50","PivotOutstanding")
            pt3.PivotFields("Emiten").Orientation=xlRowField
            pt3.PivotFields("Kelas GSS").Orientation=xlColumnField
            df=pt3.AddDataField(pt3.PivotFields("Outstanding (Rp M)"),
                                "Total Outstanding (Rp M)",xlSum)
            df.NumberFormat="#,##0.0"
            pt3.ColumnGrand=True; pt3.RowGrand=True
            pt3.TableStyle2="PivotStyleMedium2"
            print("    OK Pivot 3 berhasil")
        except Exception as e:
            print(f"    ! Pivot 3 gagal: {e}")

        wb.Save()
        print("  File tersimpan dengan pivot table")

    except Exception as e:
        print(f"  ! Error COM: {e}")
    finally:
        if xl:
            try: xl.Quit()
            except: pass


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=== PEMBUATAN EXCEL INTERAKTIF AN GSS ===\n")
    census    =_load("market_census.csv")
    comparison=_load("comparison_results.csv")
    sectors   =_load("sector_decomposition.csv")

    wb=openpyxl.Workbook()
    wb.remove(wb.active)

    print("[1/2] Membangun workbook (openpyxl)...")
    sheet_nav(wb,census,comparison,sectors)
    sheet_ringkasan(wb,census,comparison,sectors)
    sheet_sensus(wb,census)
    sheet_daftar(wb)
    sheet_dekomposisi(wb,sectors)
    sheet_evaluasi(wb,comparison)
    sheet_gold(wb,comparison)
    sheet_taksonomi(wb)
    sheet_sbn(wb)
    sheet_kat_expanded(wb,sectors)
    sheet_pivot_placeholder(wb)

    # Aktifkan sheet navigasi saat buka
    wb.active=wb["0. Navigasi"]

    wb.save(OUT_PATH)
    print(f"\n  Tersimpan: {OUT_PATH}")

    add_pivot_tables(OUT_PATH)
    print(f"\nSelesai.\n  {OUT_PATH}")


if __name__=="__main__":
    main()
