---
name: run-green-bond-classifier
description: Build, launch, screenshot, and drive the GSS bond-classifier Streamlit web app (webapp/app.py) and its classifier engine. Use when asked to run, start, serve, smoke-test, screenshot, or demo the EBUS GSS classification app, or to invoke the ML/rule classifier directly.
---

# Run the GSS EBUS classifier

Streamlit web app ([webapp/app.py](webapp/app.py)) over a taxonomy-grounded
classifier ([classifier/](classifier/)) that sorts Indonesian corporate bonds
(EBUS) into Green / Social / Sustainability classes. Three tabs: classify a
prospectus, market statistics, model evaluation. Everything runs **locally and
free** (sentence-transformers + PyMuPDF) — no paid API.

The app is driven programmatically by
[.claude/skills/run-green-bond-classifier/driver.py](.claude/skills/run-green-bond-classifier/driver.py):
it launches Streamlit headless, waits for health, then drives the UI with
Playwright using the **system Edge/Chrome** (`channel=`, so no
`playwright install` download), walks all three tabs, runs one real
classification, and saves a screenshot per step.

All paths below are relative to the repo root (the `<unit>` dir). On Windows use
the `py`/`python` on PATH; the driver re-launches Streamlit with the same
interpreter.

## Prerequisites

Python 3.14 with these already importable (verified present this session):
`streamlit` 1.58, `sentence-transformers` 5.6, `pymupdf` 1.27, `pandas` 3.0,
`playwright` (Python). If a fresh machine is missing them:

```bash
pip install streamlit sentence-transformers pymupdf pandas playwright
```

No `playwright install` is needed — the driver uses the system browser via
`channel="msedge"` (fallback `"chrome"`). Both are present on Windows. The
first classification downloads the MiniLM model (~120 MB) once from HuggingFace,
then caches it.

## Run (agent path) — the driver

From the repo root:

```bash
python .claude/skills/run-green-bond-classifier/driver.py
```

What it does and prints:

```
Streamlit healthy on http://localhost:8765

=== STEP RESULTS ===
  [PASS] load app + tab 1 visible
  [PASS] classify pasted green UoP -> positive GSS + sectors
  [PASS] tab 2 market stats rendered
  [PASS] tab 3 evaluation rendered

Screenshots: .../_run_shots
PASS
```

Exit code 0 = every step passed. Screenshots land in `_run_shots/` at the repo
root (gitignored): `1_tab_klasifikasi.png`, `2_hasil_klasifikasi.png`,
`3_statistik_pasar.png`, `4_evaluasi_model.png`. **Open
`2_hasil_klasifikasi.png`** to confirm a real verdict (e.g. "Sustainability —
Keberlanjutan · Keyakinan 96%" with a use-of-proceeds sector table).

Useful flags:

```bash
# choose port + screenshot dir
python .claude/skills/run-green-bond-classifier/driver.py --port 8765 --shot-dir ./_shots
# leave the server running afterward for manual poking
python .claude/skills/run-green-bond-classifier/driver.py --keep
```

First run is slow (~30–120 s) because of the model warm-up; the driver waits up
to 120 s for the classification result, so let it finish.

## Direct invocation — classifier without the UI

Most engine PRs touch [classifier/](classifier/), not the Streamlit layer.
Import and call directly (no server, no browser):

```bash
python -c "from classifier.ml_engine import classify_ml; r = classify_ml('Penggunaan dana untuk proyek pembangkit listrik tenaga surya dan energi terbarukan.', issuer=None); print(r.gss_class.value, r.is_gss, r.top_env[:2])"
# -> green True [('renewable_energy', 0.90...), ('energy_efficiency', 0.68...)]
```

To re-measure precision/recall on the gold set (walks the curated prospektus
corpus, prints a confusion matrix for both engines, writes
`data/comparison_results.csv`):

```bash
python evaluation/compare_engines.py
# ML SEMANTIC: Precision 1.00 / Recall 1.00 / F1 1.00 (TP/FP/FN/TN = 27/0/0/66)
```

## Run (human path)

```bash
streamlit run webapp/app.py
```

Opens a browser tab at `http://localhost:8501`. Fine for hands-on use; useless
for an automated/headless agent (it just waits for a human). Ctrl-C to stop.

## Gotchas

- **Playwright browsers are NOT installed** — `p.chromium.launch()` fails with
  "Executable doesn't exist". The driver deliberately uses
  `channel="msedge"`/`"chrome"` to borrow the system browser instead. Do not
  "fix" this by running `playwright install` unless you actually want the
  download.
- **First classification is slow and can look hung** — it downloads/loads the
  ~120 MB MiniLM model. The driver's result wait is 120 s for this reason. The
  Streamlit `@st.cache_resource` keeps it warm for later runs in the same
  server process.
- **The result banner, not the page caption, is the success signal.** The word
  "Green" appears in the static header caption regardless of any classification,
  so the driver asserts on `"Sektor eligible terpenuhi"` + `"Keyakinan:"` +
  absence of `"Non-GSS"` instead.
- **Windows long paths (>260 char) plague PDF reads** — the corpus path is deep.
  `classifier/engine.py:read_pdf_text` already retries with the `\\?\` prefix;
  the web upload path (`read_pdf_bytes`) sidesteps it entirely by reading bytes
  in memory.
- **A scanned/image-only PDF yields empty text** → the app shows "PDF tidak
  menghasilkan teks … perlu OCR" rather than crashing. Expected, not a bug.
- **Tab 3 (Evaluasi) is empty until `compare_engines.py` has been run** — it
  reads `data/comparison_results.csv`. That file is committed, so it shows by
  default; if you wipe it, the tab prints an "info" prompt instead of metrics.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FAIL: Streamlit did not become healthy in time` | Port already in use — re-run with `--port <free>`. Or Streamlit crashed on import: run `python -c "import webapp.app"` from root to see the real traceback. |
| `RuntimeError: No system Edge/Chrome usable by Playwright` | Neither Edge nor Chrome found. Install one, or run `python -m playwright install chromium` and edit `_new_browser` to drop the `channel=`. |
| Classification step times out | Model still downloading on a slow link; re-run once the cache is populated, or pre-warm with the direct-invocation one-liner above. |
| `ModuleNotFoundError: classifier` when running the driver | Run from the repo root (the driver sets `cwd` for Streamlit, but the direct-invocation commands rely on your shell being at root). |
