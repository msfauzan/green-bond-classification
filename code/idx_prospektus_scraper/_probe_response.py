"""Probe struktur respons API announcement IDX dan cari filter kode emiten."""
import json, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

FETCH_JS = """
const url = arguments[0];
const done = arguments[arguments.length - 1];
fetch(url, {
  headers: {'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest'},
  credentials: 'include'
})
  .then(r => r.text())
  .then(t => done({ok: true, body: t}))
  .catch(e => done({ok: false, error: String(e)}));
"""

BASE_API = "https://www.idx.co.id/primary/NewsAnnouncement/GetAllAnnouncement"

opts = Options()
opts.add_argument("--start-maximized")
opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
driver = webdriver.Chrome(options=opts)
driver.set_script_timeout(60)

driver.get("https://www.idx.co.id/id/berita/pengumuman/")
print("Tunggu Cloudflare...")
time.sleep(10)

# Test 1: query biasa dengan keyword prospektus
url1 = f"{BASE_API}?keywords=prospektus&pageNumber=1&pageSize=3&lang=id"
print(f"\n[1] Query: {url1}")
res = driver.execute_async_script(FETCH_JS, url1)
if res and res.get("ok"):
    data = json.loads(res["body"])
    print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
else:
    print("GAGAL:", res)

# Test 2: coba tambah filter stockCode
url2 = f"{BASE_API}?keywords=prospektus&stockCode=PNMP&pageNumber=1&pageSize=3&lang=id"
print(f"\n[2] Query dengan stockCode=PNMP: {url2}")
res2 = driver.execute_async_script(FETCH_JS, url2)
if res2 and res2.get("ok"):
    data2 = json.loads(res2["body"])
    anns = data2.get("Announcements") or data2.get("announcements") or []
    print(f"  Result count: {data2.get('TotalData') or data2.get('totalData') or '?'}")
    print(f"  Announcements: {len(anns)}")
    if anns:
        print("  Sample keys:", list(anns[0].keys()))
        print("  Sample[0]:", json.dumps(anns[0], indent=4, ensure_ascii=False)[:1500])
else:
    print("GAGAL:", res2)

driver.quit()
