import json, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

opts = Options()
opts.add_argument("--start-maximized")
opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
driver = webdriver.Chrome(options=opts)

driver.get("https://www.idx.co.id/id/berita/pengumuman/")
time.sleep(10)

# Isi field 'Kata kunci...'
try:
    kw_input = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Kata kunci..."]')
    kw_input.clear()
    kw_input.send_keys("prospektus")
    time.sleep(1)
    kw_input.send_keys(Keys.RETURN)
    print("[OK] Keyword diisi: prospektus, Enter dikirim")
except Exception as e:
    print(f"[!] Gagal isi keyword: {e}")

time.sleep(8)

# Coba klik tombol Cari / Search jika ada
try:
    btns = driver.find_elements(By.TAG_NAME, "button")
    for b in btns:
        txt = (b.text or "").lower()
        if "cari" in txt or "search" in txt or "filter" in txt:
            print(f"[i] Klik tombol: '{b.text}'")
            b.click()
            time.sleep(5)
            break
except Exception as e:
    print(f"[!] Gagal klik: {e}")

# Baca semua input
inputs = driver.find_elements(By.TAG_NAME, "input")
print(f"\nInput fields ({len(inputs)}):")
for inp in inputs:
    ph = inp.get_attribute("placeholder") or ""
    tp = inp.get_attribute("type") or ""
    nm = inp.get_attribute("name") or ""
    val = inp.get_attribute("value") or ""
    print(f"  type={tp} name={nm} placeholder={ph} value={val}")

# Baca network log
logs = driver.get_log("performance")
found = []
for entry in logs:
    try:
        msg = json.loads(entry["message"])["message"]
        if msg.get("method") == "Network.responseReceived":
            resp = msg.get("params", {}).get("response", {})
            url = resp.get("url", "")
            mime = (resp.get("mimeType") or "").lower()
            if "json" in mime and "idx.co.id" in url.lower():
                found.append(url)
    except Exception:
        pass

print("\n=== JSON API calls setelah search ===")
for u in sorted(set(found)):
    print(" ", u)

# Juga cari XHR requests (request sent)
reqs = []
for entry in logs:
    try:
        msg = json.loads(entry["message"])["message"]
        if msg.get("method") == "Network.requestWillBeSent":
            req = msg.get("params", {}).get("request", {})
            url = req.get("url", "")
            if "idx.co.id" in url and ("announcement" in url.lower() or
               "pengumuman" in url.lower() or "secondary" in url.lower() or
               "primary" in url.lower()):
                reqs.append(url)
    except Exception:
        pass

print("\n=== Semua XHR requests ke idx.co.id ===")
for u in sorted(set(reqs)):
    print(" ", u)

driver.quit()
