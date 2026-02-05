"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SCRAPER PDF PROSPEKTUS IDX                                          ║
║           Bank Indonesia - DSta-DSMF - Green Bond Classification             ║
║                                                                              ║
║  Fungsi:                                                                     ║
║  - Download PDF prospektus dari IDX                                          ║
║  - Rename file dengan format: YYYYMMDD_KODE_NamaFile.pdf                     ║
║  - Hindari duplikasi (skip file yang sudah ada)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = r"d:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
DOWNLOAD_DIR = os.path.join(BASE_DIR, "Prospektus_Downloaded")

IDX_URL = "https://www.idx.co.id/id/berita/pengumuman"
SEARCH_KEYWORD = "prospektus"
MAX_PAGES = 5  # Default halaman yang di-scrape


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_filename(filename):
    """Remove karakter tidak valid dari nama file."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Ganti multiple spaces dengan single space
    filename = re.sub(r'\s+', ' ', filename).strip()
    # Batasi panjang filename
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def setup_chrome_driver():
    """Setup Chrome WebDriver dengan auto-download PDF."""
    print("🔧 Setting up Chrome WebDriver...")
    
    chrome_options = Options()
    
    # Download preferences - auto download PDF tanpa preview
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "plugins.plugins_disabled": ["Chrome PDF Viewer"],
        "profile.default_content_settings.popups": 0,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    print("✅ Chrome WebDriver ready!")
    return driver


def wait_for_download(timeout=60):
    """Tunggu sampai download selesai (tidak ada file .crdownload)."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        downloading = False
        for f in os.listdir(DOWNLOAD_DIR):
            if f.endswith('.crdownload') or f.endswith('.tmp'):
                downloading = True
                break
        if not downloading:
            time.sleep(1)
            return True
        time.sleep(1)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCRAPER
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_prospektus(max_pages=MAX_PAGES):
    """
    Scrape PDF prospektus dari IDX.
    
    Args:
        max_pages: Jumlah halaman IDX yang akan di-scrape
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║           SCRAPER PDF PROSPEKTUS IDX                                          ║
    ║           Bank Indonesia - DSta-DSMF - Green Bond Classification             ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📂 Download Directory: {DOWNLOAD_DIR}")
    print(f"📄 Max Pages: {max_pages}")
    print("=" * 70)
    
    # Buat folder download jika belum ada
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    driver = None
    downloaded_count = 0
    skipped_count = 0
    
    try:
        driver = setup_chrome_driver()
        driver.get(IDX_URL)
        time.sleep(3)
        
        wait = WebDriverWait(driver, 15)
        
        # ===== SEARCH =====
        print(f"\n🔍 Searching for '{SEARCH_KEYWORD}'...")
        try:
            search_input = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input[placeholder='Kata kunci...']")
            ))
            search_input.clear()
            search_input.send_keys(SEARCH_KEYWORD)
            time.sleep(1)
            search_input.send_keys(Keys.ENTER)
            time.sleep(3)
            print("✅ Search completed!")
        except Exception as e:
            print(f"⚠️ Search issue: {e}")
        
        # ===== LOOP PAGES =====
        current_page = 1
        while current_page <= max_pages:
            print(f"\n{'='*70}")
            print(f"📄 PAGE {current_page}/{max_pages}")
            print("="*70)
            
            time.sleep(2)
            
            # Find announcement cards
            cards = driver.find_elements(By.CSS_SELECTOR, "div.attach-card")
            print(f"📋 Found {len(cards)} announcements")
            
            if len(cards) == 0:
                print("⚠️ No more announcements found")
                break
            
            # ===== LOOP CARDS =====
            for idx, card in enumerate(cards, 1):
                try:
                    # Get title
                    title_elem = card.find_element(By.CSS_SELECTOR, "h6.title")
                    title = title_elem.text.strip()
                    
                    # Get date from <time> element
                    try:
                        date_elem = card.find_element(By.TAG_NAME, "time")
                        date_text = date_elem.text.strip()
                        # Convert "02 Feb 2026 17:22:44" -> "20260202"
                        date_parts = date_text.split()
                        if len(date_parts) >= 3:
                            day = date_parts[0].zfill(2)
                            month_map = {
                                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
                                'Okt': '10', 'Mei': '05', 'Agu': '08', 'Des': '12'
                            }
                            month = month_map.get(date_parts[1], '01')
                            year = date_parts[2]
                            date_clean = f"{year}{month}{day}"
                        else:
                            date_clean = datetime.now().strftime('%Y%m%d')
                    except:
                        date_clean = datetime.now().strftime('%Y%m%d')
                    
                    # Extract stock code from title [XXXX]
                    code_match = re.search(r'\[([A-Z]{4})\s*\]', title)
                    stock_code = code_match.group(1) if code_match else "UNKNOWN"
                    
                    print(f"\n[{idx}] {stock_code} | {date_clean} | {title[:50]}...")
                    
                    # Find PDF links
                    pdf_links = card.find_elements(By.CSS_SELECTOR, "a[href$='.pdf']")
                    
                    for link in pdf_links:
                        href = link.get_attribute('href')
                        link_text = link.text.strip()
                        
                        if not href:
                            continue
                        
                        # Create proper filename
                        original_name = link_text if link_text else os.path.basename(href)
                        filename = sanitize_filename(f"{date_clean}_{stock_code}_{original_name}")
                        if not filename.lower().endswith('.pdf'):
                            filename += '.pdf'
                        
                        filepath = os.path.join(DOWNLOAD_DIR, filename)
                        
                        # Skip if already exists
                        if os.path.exists(filepath):
                            print(f"    ⏭️  Skip (exists): {filename[:60]}...")
                            skipped_count += 1
                            continue
                        
                        # Download
                        print(f"    📥 Downloading: {filename[:60]}...")
                        
                        files_before = set(os.listdir(DOWNLOAD_DIR))
                        
                        # Open PDF in new tab (triggers download)
                        driver.execute_script(f"window.open('{href}', '_blank');")
                        time.sleep(2)
                        
                        # Close new tab and switch back
                        if len(driver.window_handles) > 1:
                            driver.switch_to.window(driver.window_handles[-1])
                            time.sleep(1)
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                        
                        # Wait for download
                        if wait_for_download(30):
                            files_after = set(os.listdir(DOWNLOAD_DIR))
                            new_files = files_after - files_before
                            
                            for new_file in new_files:
                                if new_file.endswith('.pdf'):
                                    new_file_path = os.path.join(DOWNLOAD_DIR, new_file)
                                    
                                    # Rename to our filename
                                    if new_file != filename:
                                        try:
                                            os.rename(new_file_path, filepath)
                                            print(f"    ✅ Saved: {filename[:60]}...")
                                        except Exception as e:
                                            print(f"    ⚠️ Rename failed: {e}")
                                            filepath = new_file_path
                                    else:
                                        print(f"    ✅ Downloaded: {new_file[:60]}...")
                                    
                                    downloaded_count += 1
                        
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"    ❌ Error: {str(e)[:50]}")
                    continue
            
            # ===== NEXT PAGE =====
            if current_page < max_pages:
                try:
                    next_btn = driver.find_element(By.CSS_SELECTOR, 
                        "button.btn-arrow.--next:not([disabled])")
                    next_btn.click()
                    print(f"\n➡️ Going to page {current_page + 1}...")
                    time.sleep(3)
                    current_page += 1
                except NoSuchElementException:
                    print("\n⚠️ No more pages available")
                    break
                except Exception as e:
                    print(f"\n⚠️ Could not go to next page: {e}")
                    break
            else:
                current_page += 1
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        if driver:
            driver.quit()
            print("\n🔒 Browser closed")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"✅ Downloaded: {downloaded_count} files")
    print(f"⏭️  Skipped (already exists): {skipped_count} files")
    print(f"📂 Location: {DOWNLOAD_DIR}")
    print("="*70)
    
    return downloaded_count, skipped_count


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape PDF Prospektus dari IDX")
    parser.add_argument("--pages", type=int, default=MAX_PAGES, 
                        help=f"Jumlah halaman yang di-scrape (default: {MAX_PAGES})")
    
    args = parser.parse_args()
    
    scrape_prospektus(max_pages=args.pages)
