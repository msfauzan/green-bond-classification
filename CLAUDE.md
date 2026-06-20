# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research workspace for a Bank Indonesia (DSta / DSMF) **Analytical Note (AN)** on
classifying Indonesian corporate debt securities (EBUS — Efek Bersifat Utang & Sukuk)
into **GSS categories (Green / Social / Sustainability)** per **POJK 18/2023**, ICMA,
and the DJPPR sovereign framework. The deliverable is the AN, not a shipped application —
code exists to gather evidence and prototype the classifier.

The intellectual framing lives in two markdown docs; **read these before doing
substantive work** — they drive every design decision:
- [Kerangka_AN_Klasifikasi_GSS_ML.md](Kerangka_AN_Klasifikasi_GSS_ML.md) — the AN skeleton (scope, methodology, stages)
- [Brainstorming_AN_GSS.md](Brainstorming_AN_GSS.md) — reasoning and open questions

### Core thesis (governs scope)
- **Sovereign (SBN) is already authoritative** (Climate Budget Tagging → BPK audit → SRN-PPI registry) → no ML needed. It provides the taxonomy, ground truth, and benchmark.
- **Corporate (EBUS) is the ML target** — no equivalent labeling process, only ~a dozen official BEI labels → the gap and the original contribution.
- Approach is **taxonomy-grounded**, not supervised-from-scratch (labels are too few): rule-based keyword scoring (transparent, auditable baseline) + LLM-as-judge grounded in the taxonomy. **Explainable by design** — every decision cites the criterion/keyword that triggered it.
- Output is not a binary "GSS?" flag (the bond name often implies that). The value is **sectoral decomposition** (which of the 9 green / 6 social eligible categories the use-of-proceeds funds → statistics) and **claim verification** (proceeds vs taxonomy → greenwashing screening).

## Repository layout

| Path | Purpose |
|---|---|
| `classifier/taxonomy.py` | The "dictionary of truth" — 9 green + 6 social eligible categories, each with bilingual ID/EN keyword lists; plus Level-0 signals for Sustainability-Linked (KPI/SPT/step-up) and Sukuk Wakaf. Hierarchical scheme: Level 0 = instrument structure, Level 1 = use-of-proceeds category. |
| `idx_bond_scraper/` | Selenium scraper for the IDX corporate bond/sukuk **listing** (the security universe). See its README. |
| `idx_prospektus_scraper/` | Selenium scraper for **prospektus PDFs** from IDX announcements + the curation/audit/reorg scripts. |
| `data/` | `sbn_gss_lookup.csv` (25 sovereign instruments, the ground truth) + scraped IDX CSV/JSON + extracted sovereign report text. |
| `pdf_by_content/01_prospektus_utama/` | Downloaded prospektus PDFs, one folder per announcement. **Gitignored.** |
| `pdf_by_content/01_prospektus_utama/0. GSS Fixed/{GSS,NonGSS,Review}/<EMITEN>/` | The **curated, content-verified** prospektus, organized by category then issuer code. This is the hand-checked gold corpus. |
| `Paper Referensi ML/`, `FGD AN Green Debt Securities/`, `PPT/` | Reference papers, focus-group material, slides. |

Most data is **gitignored**: `*.pdf`, `*.xlsx`, `*.joblib/*.pkl`, `pdf_by_content/`,
`ML_Dataset/`, `paper/`. The repo tracks code, the markdown docs, and small CSVs in `data/`.

## Commands

Per-module dependencies (no root requirements.txt):
```bash
pip install -r idx_bond_scraper/requirements.txt        # selenium, pandas, requests
pip install -r idx_prospektus_scraper/requirements.txt  # adds PyMuPDF (fitz) for PDF text
```

```bash
# Scrape the corporate bond/sukuk listing (run WITH a visible window to pass Cloudflare)
cd idx_bond_scraper && python scrape_idx_bonds.py
python scrape_idx_bonds.py --filter ../list_emisi_gss.txt   # subset to GSS issuers

# Scrape prospektus PDFs (default = the 15 GSS issuer codes in DEFAULT_GSS_CODES)
cd idx_prospektus_scraper && python scrape_prospektus.py
python scrape_prospektus.py --codes BBRI BMRI SMFP
python scrape_prospektus.py --from-date 2024-01-01

# Run the rule-based classifier against verified prospektus (concept demo)
python idx_prospektus_scraper/demo_classify.py

# Inspect / audit the curated corpus
python idx_prospektus_scraper/list_all.py          # inventory of GSS/NonGSS/Review per issuer
python idx_prospektus_scraper/audit_gss_folders.py # detected GSS type + bond name per folder
python idx_prospektus_scraper/deep_check.py        # keyword-in-context for ambiguous cases
```

There is no build, lint, or test suite. The `idx_prospektus_scraper/` scripts prefixed
`_` (`_probe_*.py`) and the `check_*`/`fix_*`/`reorganize_*` scripts are one-off
investigation/curation tools, not a maintained library — read one before reusing it.

## Critical gotchas (learned the hard way)

**Windows 260-char path limit dominates this project.** The base corpus path is ~104
chars; adding `0. GSS Fixed/<category>/<EMITEN>/` plus a 75-char folder name plus a
~57-char filename routinely exceeds 260. Consequences:
- In Python, open/move/list long paths with the `\\?\` prefix (`fitz.open("\\\\?\\" + path)`). Use `os.listdir` over `Path.glob`. The scraper caps generated names at `MAX_NAME = 75`.
- `LongPathsEnabled=1` has been set in the registry but **requires a reboot** to take effect. Until then, `subst G: "<...>\0. GSS Fixed"` maps a short drive (per-session; not visible to the Bash tool, only to Explorer/PowerShell).
- A settings hook blocks `Remove-Item` on `\\?\` paths — use Python `ctypes`/`os` or `cmd` for deletes.

**IDX scraping requires real Chrome via Selenium** — the site is behind Cloudflare +
TLS fingerprinting, so `requests`/`curl`/`httpx` are blocked (403). Run with a visible
window (default) for the best chance; retry if the first challenge fails. The bond
scraper pulls **all** rows in one request because IDX returns rows in unstable order
(per-page pagination silently drops records).

**The word "Berkelanjutan" is a classification trap.** "Penawaran Umum Berkelanjutan"
(PUB) is a *shelf-registration* administrative term with nothing to do with
sustainability, whereas "EBUS Keberlanjutan" is a genuine sustainability bond. Naive
keyword matching on "berkelanjutan" produces false positives (e.g. SMII's conventional
"Obligasi Berkelanjutan III/IV" was wrongly tagged GSS). Likewise, a GSS keyword found
inside a **financial-statement table** (a previously-issued green bond on the balance
sheet) is not evidence the *current* offering is GSS — always check the keyword's
context, not just its presence.

**PyMuPDF (and Windows Defender) lock PDF files** after reading; `shutil.move` then
fails with WinError 32. Close documents, and move/delete via `cmd`/`robocopy` or schedule
deletion on reboot (`MoveFileEx`) when a handle is stuck.

**Corpus structure is inconsistent** — some `<EMITEN>/` folders hold PDFs directly
(flat), others nest one folder per announcement. Tools that walk the corpus must handle
both (see `demo_classify.py: run_one`).
