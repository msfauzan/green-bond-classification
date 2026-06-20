"""
Bandingkan engine rule-based (engine.py) vs ML/semantik (ml_engine.py).

Mencetak:
  - confusion matrix + P/R/F1 keduanya berdampingan
  - skor similarity per kategori untuk FP/FN ML engine (diagnosis)
  - data/comparison_results.csv (detail per dokumen)
"""
from __future__ import annotations
import os
import sys
import csv

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
sys.path.insert(0, ROOT)

from classifier.engine    import read_pdf_text, classify       as classify_rule
from classifier.ml_engine import classify_ml
from classifier.taxonomy  import GSSClass

BASE_PROS = os.path.join(ROOT, "pdf_by_content", "01_prospektus_utama")
GOLD_GSS  = os.path.join(BASE_PROS, "0. Fix GSS", "GSS")
GSS_ISSUERS = {"ARKO","BBNI","BBRI","BBTN","BJBR","BMRI",
                "BRIS","OPPM","PNMP","POLI","PPGD","SMII"}
MAX_PAGES = 50
LP = "\\\\?\\"


# --- long-path helpers -------------------------------------------------------
def _ll(path):
    for p in (path, LP + path):
        try: return os.listdir(p)
        except OSError: continue
    return []

def _isdir(p): return os.path.isdir(p) or os.path.isdir(LP + p)
def _sz(p):
    for q in (p, LP + p):
        try: return os.path.getsize(q)
        except OSError: continue
    return 0

def _all_pdfs(root):
    out = []
    for e in _ll(root):
        full = os.path.join(root, e)
        if _isdir(full): out.extend(_all_pdfs(full))
        elif e.lower().endswith(".pdf"): out.append(full)
    return out

def _largest_pdf(folder):
    pdfs = [(_sz(os.path.join(folder, f)), f)
            for f in _ll(folder) if f.lower().endswith(".pdf")]
    pdfs.sort(reverse=True)
    return os.path.join(folder, pdfs[0][1]) if pdfs else None


# --- gold set ----------------------------------------------------------------
def build_gold():
    rows = []
    for iss in sorted(_ll(GOLD_GSS)):
        ip = os.path.join(GOLD_GSS, iss)
        if not _isdir(ip): continue
        for pdf in _all_pdfs(ip):
            rows.append({"issuer": iss, "gold": "GSS", "path": pdf})
    by_issuer = {}
    for name in sorted(_ll(BASE_PROS)):
        full = os.path.join(BASE_PROS, name)
        if name == "0. Fix GSS" or not _isdir(full): continue
        parts = name.split("_"); iss = parts[1] if len(parts) >= 2 else "?"
        if iss in GSS_ISSUERS: continue
        by_issuer.setdefault(iss, []).append(full)
    for iss, folders in sorted(by_issuer.items()):
        best = max(folders, key=lambda f: _sz(_largest_pdf(f) or ""))
        pdf  = _largest_pdf(best)
        if pdf: rows.append({"issuer": iss, "gold": "NonGSS", "path": pdf})
    return rows


# --- scoring -----------------------------------------------------------------
def score(results, gold_key="gold", pred_key="pred"):
    tp=fp=fn=tn=0
    for r in results:
        g = r[gold_key]; p = r[pred_key]
        if g=="GSS" and p=="GSS":   tp+=1
        elif g!="GSS" and p=="GSS": fp+=1
        elif g=="GSS" and p!="GSS": fn+=1
        else: tn+=1
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return tp, fp, fn, tn, prec, rec, f1


# --- main --------------------------------------------------------------------
def main():
    gold = build_gold()
    n_pos = sum(1 for r in gold if r["gold"] == "GSS")
    n_neg = len(gold) - n_pos
    print(f"Gold set: {len(gold)} dok  ({n_pos} GSS, {n_neg} NonGSS)\n")
    print("Memproses PDF (ML: unduh model pertama kali ~120 MB)...\n")

    out_rows = []
    fp_ml = []; fn_ml = []

    for r in gold:
        text = read_pdf_text(r["path"], MAX_PAGES)

        rr = classify_rule(text)
        rm = classify_ml(text, issuer=r["issuer"])

        pred_rule = ("GSS" if (rr and rr.is_gss)  else "NonGSS") if rr else "SKIP"
        pred_ml   = ("GSS" if (rm and rm.is_gss)  else "NonGSS") if rm else "SKIP"

        row = {
            "issuer"    : r["issuer"],
            "gold"      : r["gold"],
            "pred_rule" : pred_rule,
            "pred_ml"   : pred_ml,
            "rule_class": rr.gss_class.value if rr else "",
            "ml_class"  : rm.gss_class.value if rm else "",
            "ml_title_gss": str(rm.title_gss) if rm else "",
            "ml_framing_body": "|".join(rm.framing_body[:2]) if rm else "",
            "ml_framing_title": rm.framing_title[0][:40] if (rm and rm.framing_title) else "",
            "ml_thresh" : rm.threshold_used   if rm else "",
            "ml_top_env": "|".join(f"{k}:{s:.2f}" for k,s in (rm.top_env or [])) if rm else "",
            "ml_top_soc": "|".join(f"{k}:{s:.2f}" for k,s in (rm.top_soc or [])) if rm else "",
            "ml_maxscore": f"{max(rm.scores.values()):.3f}" if (rm and rm.scores) else "",
            "ml_conf"   : rm.confidence if rm else "",
            "pdf"       : os.path.basename(r["path"]),
        }
        out_rows.append(row)

        if rm and pred_ml == "GSS"    and r["gold"] == "NonGSS": fp_ml.append({**r, "rm": rm})
        if rm and pred_ml == "NonGSS" and r["gold"] == "GSS":   fn_ml.append({**r, "rm": rm})

    res_rule = [{"gold": r["gold"], "pred": r["pred_rule"]} for r in out_rows if r["pred_rule"] != "SKIP"]
    res_ml   = [{"gold": r["gold"], "pred": r["pred_ml"]}   for r in out_rows if r["pred_ml"]   != "SKIP"]

    tp_r,fp_r,fn_r,tn_r,pr_r,re_r,f1_r = score(res_rule)
    tp_m,fp_m,fn_m,tn_m,pr_m,re_m,f1_m = score(res_ml)

    print("=" * 70)
    print(f"  {'':30} {'RULE-BASED':>15}  {'ML SEMANTIC':>15}")
    print("=" * 70)
    print(f"  {'TP / FP / FN / TN':30} {tp_r}/{fp_r}/{fn_r}/{tn_r:>5}  {tp_m}/{fp_m}/{fn_m}/{tn_m:>5}")
    print(f"  {'Precision':30} {pr_r:>15.2f}  {pr_m:>15.2f}")
    print(f"  {'Recall':30} {re_r:>15.2f}  {re_m:>15.2f}")
    print(f"  {'F1':30} {f1_r:>15.2f}  {f1_m:>15.2f}")
    print()

    # distribusi sub-kelas ML pada positif benar
    from collections import Counter
    sub = Counter(r["ml_class"] for r in out_rows if r["gold"]=="GSS" and r["pred_ml"]=="GSS")
    print("  Sub-kelas ML (TP):")
    for k,v in sub.most_common(): print(f"     {k:28} {v}")

    if fn_ml:
        print(f"\n  FALSE NEGATIVE ML ({len(fn_ml)}):")
        for c in fn_ml:
            rm = c["rm"]
            top = max(rm.scores.items(), key=lambda x: x[1]) if rm.scores else ("—", 0)
            print(f"     {c['issuer']:6} title={rm.title_gss}  "
                  f"max={top[1]:.2f}({top[0]})  thr={rm.threshold_used:.2f}  "
                  f"{os.path.basename(c['path'])[:45]}")

    if fp_ml:
        print(f"\n  FALSE POSITIVE ML ({len(fp_ml)}):")
        for c in fp_ml:
            rm = c["rm"]
            top_e = rm.top_env[:2]; top_s = rm.top_soc[:2]
            hits = "|".join(f"{k}:{s:.2f}" for k,s in (top_e+top_s)[:3])
            print(f"     {c['issuer']:6} title={rm.title_gss}  "
                  f"[{hits}]  thr={rm.threshold_used:.2f}  {os.path.basename(c['path'])[:35]}")

    out_csv = os.path.join(ROOT, "data", "comparison_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\n  Detail -> {out_csv}")


if __name__ == "__main__":
    main()
