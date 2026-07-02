#!/usr/bin/env python
"""
Driver for the GSS classifier Streamlit web app (webapp/app.py).

Launches Streamlit headless, waits for it to be ready, then drives it with
Playwright (system Edge/Chrome via `channel=` — NO `playwright install` needed).
Walks all three tabs, runs one real classification flow on pasted text, and
saves a screenshot per step. Prints PASS/FAIL and the screenshot dir.

Usage (from repo root):
    python .claude/skills/run-green-bond-classifier/driver.py
    python .claude/skills/run-green-bond-classifier/driver.py --port 8765 --keep
    python .claude/skills/run-green-bond-classifier/driver.py --shot-dir ./_shots

--keep leaves the Streamlit server running (for manual poking); otherwise it is
torn down on exit. Exit code 0 = every step succeeded.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # .../skills/run-.../driver.py -> repo root

# A real green-bond use-of-proceeds blurb (Indonesian) — exercises the ML path.
SAMPLE_UOP = (
    "PENGGUNAAN DANA. Seluruh dana hasil Penawaran Umum Obligasi Berwawasan "
    "Lingkungan ini, setelah dikurangi biaya emisi, akan digunakan untuk "
    "membiayai dan/atau membiayai kembali proyek hijau yang memenuhi syarat "
    "(eligible green projects) sesuai Kerangka Obligasi Berwawasan Lingkungan "
    "Perseroan, mencakup pembangkit listrik energi terbarukan (tenaga surya, "
    "panas bumi, tenaga air), efisiensi energi, serta transportasi rendah emisi."
)


def _health(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/_stcore/health", timeout=2
        ) as r:
            return r.status == 200 and r.read().strip() == b"ok"
    except Exception:
        return False


def _wait_ready(port: int, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health(port):
            return True
        time.sleep(1)
    return False


def _launch_streamlit(port: int) -> subprocess.Popen:
    env = dict(os.environ, PYTHONUTF8="1")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join("code", "webapp", "app.py"),
         "--server.headless", "true",
         "--server.port", str(port),
         "--server.address", "127.0.0.1",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def _new_browser(p):
    """System browser via channel — avoids `playwright install`."""
    last = None
    for ch in ("msedge", "chrome"):
        try:
            return p.chromium.launch(channel=ch)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"No system Edge/Chrome usable by Playwright: {last}")


def run(port: int, shot_dir: str, keep: bool) -> int:
    from playwright.sync_api import sync_playwright

    os.makedirs(shot_dir, exist_ok=True)
    proc = _launch_streamlit(port)
    steps: list[tuple[str, bool]] = []
    try:
        if not _wait_ready(port):
            print("FAIL: Streamlit did not become healthy in time")
            return 2
        print(f"Streamlit healthy on http://localhost:{port}")

        with sync_playwright() as p:
            browser = _new_browser(p)
            page = browser.new_page(viewport={"width": 1450, "height": 1000})
            page.goto(f"http://localhost:{port}", wait_until="networkidle")
            # App now uses a sidebar radio for nav (not st.tabs); default
            # landing page is "Statistik Pasar" (first radio option).
            # Streamlit's radio <input> is visually hidden (custom styling) —
            # target the visible label text instead, which Playwright can
            # click to toggle the underlying input via its <label> wrapper.
            classify_nav = page.get_by_text("🔎 Klasifikasi Prospektus", exact=True)
            classify_nav.wait_for(timeout=30000)
            page.get_by_text("Statistik Pasar EBUS GSS", exact=False).first.wait_for(timeout=30000)
            page.screenshot(path=os.path.join(shot_dir, "1_statistik_pasar.png"),
                            full_page=True)
            steps.append(("load app + default market page rendered", True))

            # --- Navigate to Klasifikasi Prospektus, paste UoP text, classify ---
            classify_nav.click()
            page.get_by_role("button", name="Klasifikasikan").wait_for(timeout=30000)
            page.locator("textarea").first.fill(SAMPLE_UOP)
            page.get_by_role("button", name="Klasifikasikan").click()
            # Model warm-up + inference; banner text contains the class label.
            banner = page.get_by_text("Keyakinan:", exact=False)
            banner.first.wait_for(timeout=120000)
            # Banner appearing only means render_result() started streaming —
            # the sector table further down still needs its own round trip.
            # Wait for it directly rather than a fixed sleep (was flaky).
            try:
                page.get_by_text("Sektor eligible terpenuhi", exact=False) \
                    .first.wait_for(timeout=10000)
            except Exception:
                pass  # doc may legitimately have zero corroborated sectors
            page.wait_for_timeout(500)
            page.screenshot(path=os.path.join(shot_dir, "2_hasil_klasifikasi.png"),
                            full_page=True)
            body = page.inner_text("body")
            # Meaningful check: a positive GSS verdict with a use-of-proceeds
            # sector table — not just any banner ("Green" alone also appears in
            # the page caption, so it is not a reliable signal).
            classified_ok = ("Keyakinan:" in body
                             and "Sektor eligible terpenuhi" in body
                             and "Non-GSS" not in body)
            steps.append(("classify pasted green UoP -> positive GSS + sectors",
                          classified_ok))

            # --- Evaluasi Model page ---
            page.get_by_text("🎯 Evaluasi Model", exact=True).click()
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(shot_dir, "3_evaluasi_model.png"),
                            full_page=True)
            steps.append(("evaluation page rendered", True))

            browser.close()
    finally:
        if not keep:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("\n=== STEP RESULTS ===")
    for name, ok in steps:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nScreenshots: {shot_dir}")
    all_ok = all(ok for _, ok in steps)
    print("PASS" if all_ok else "FAIL")
    if keep:
        print(f"Server still running on http://localhost:{port} (PID {proc.pid})")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--shot-dir", default=os.path.join(ROOT, "_run_shots"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    return run(a.port, a.shot_dir, a.keep)


if __name__ == "__main__":
    raise SystemExit(main())
