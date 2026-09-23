"""
AgriN RAG — PDF Corpus Downloader
Downloads NIPHM IPM packages, ICAR advisories, and CIBRC data.
Run once to populate rag/data/pdfs/

Usage:
    python rag/download_pdfs.py
"""

import os
import time
import requests

PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

# ── NIPHM IPM Packages (crop-wise pest/disease management) ──────────────
# These map to PlantVillage crops in the disease model
NIPHM_BASE = "https://niphm.gov.in/IPMPackages"
NIPHM_PDFS = {
    "ipm_tomato.pdf": f"{NIPHM_BASE}/Tomato.pdf",
    "ipm_rice.pdf": f"{NIPHM_BASE}/Rice.pdf",
    "ipm_potato.pdf": f"{NIPHM_BASE}/Potato.pdf",
    "ipm_maize.pdf": f"{NIPHM_BASE}/Maize.pdf",
    "ipm_grape.pdf": f"{NIPHM_BASE}/Grape.pdf",
    "ipm_apple.pdf": f"{NIPHM_BASE}/Apple.pdf",
    "ipm_cotton.pdf": f"{NIPHM_BASE}/Cotton.pdf",
    "ipm_soybean.pdf": f"{NIPHM_BASE}/Soybean.pdf",
    "ipm_citrus.pdf": f"{NIPHM_BASE}/Citrus.pdf",        # covers orange
    "ipm_chilli.pdf": f"{NIPHM_BASE}/Chilli.pdf",         # covers pepper/bell pepper
    "ipm_strawberry.pdf": f"{NIPHM_BASE}/Strawberry.pdf",
    "ipm_peach.pdf": f"{NIPHM_BASE}/Peach.pdf",
    "ipm_cherry.pdf": f"{NIPHM_BASE}/Cherry.pdf",
    "ipm_sugarcane.pdf": f"{NIPHM_BASE}/Sugarcane.pdf",
    "ipm_wheat.pdf": f"{NIPHM_BASE}/Wheat.pdf",
    "ipm_groundnut.pdf": f"{NIPHM_BASE}/Groundnut.pdf",
    "ipm_mustard.pdf": f"{NIPHM_BASE}/Mustard.pdf",
    "ipm_lentil.pdf": f"{NIPHM_BASE}/Lentil.pdf",
    "ipm_chickpea.pdf": f"{NIPHM_BASE}/Chickpea.pdf",
    "ipm_pigeonpea.pdf": f"{NIPHM_BASE}/Pigeonpea.pdf",
    "ipm_banana.pdf": f"{NIPHM_BASE}/Banana.pdf",
    "ipm_mango.pdf": f"{NIPHM_BASE}/Mango.pdf",
    "ipm_onion.pdf": f"{NIPHM_BASE}/Onion.pdf",
    "ipm_brinjal.pdf": f"{NIPHM_BASE}/Brinjal.pdf",
    "ipm_okra.pdf": f"{NIPHM_BASE}/Okra.pdf",
    "ipm_cabbage.pdf": f"{NIPHM_BASE}/Cabbage.pdf",
    "ipm_cauliflower.pdf": f"{NIPHM_BASE}/Cauliflower.pdf",
}

# ── TNAU Crop Production Guides ──────────────────────────────────────────
TNAU_PDFS = {
    "tnau_agriculture_cpg.pdf": "https://agritech.tnau.ac.in/pdf/AGRICULTURE.pdf",
    "tnau_horticulture_cpg.pdf": "https://agritech.tnau.ac.in/pdf/HORTICULTURE.pdf",
    "tnau_ipm_procedures.pdf": "https://agritech.tnau.ac.in/crop_protection/pdf/opproc_IPM.pdf",
}

# ── PAU Package of Practices (Punjab — Kharif & Rabi) ───────────────────
PAU_PDFS = {
    "pau_kharif_pop.pdf": "https://pau.edu/content/ccil/pf/pp_kharif.pdf",
    "pau_rabi_pop.pdf": "https://pau.edu/content/ccil/pf/pp_rabi.pdf",
}

# ── ICAR Seasonal Advisory ───────────────────────────────────────────────
ICAR_PDFS = {
    "icar_kharif_advisory_2025.pdf": "https://icar.org.in/sites/default/files/Circulars/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf",
}

ALL_PDFS = {**NIPHM_PDFS, **TNAU_PDFS, **PAU_PDFS, **ICAR_PDFS}

HEADERS = {
    "User-Agent": "AgriN/3.0 (agricultural-advisory-research-prototype)"
}


def download_pdf(filename: str, url: str) -> bool:
    filepath = os.path.join(PDF_DIR, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"  ✓ {filename} already exists ({size_mb:.1f} MB)")
        return True

    try:
        print(f"  ↓ Downloading {filename}...")
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        resp.raise_for_status()

        if len(resp.content) < 1000:
            print(f"  ✗ {filename} — response too small ({len(resp.content)} bytes), likely error page")
            return False

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet" not in content_type:
            print(f"  ⚠ {filename} — unexpected Content-Type: {content_type}, saving anyway")

        with open(filepath, "wb") as f:
            f.write(resp.content)

        size_mb = len(resp.content) / 1e6
        print(f"  ✓ {filename} downloaded ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"  ✗ {filename} — HTTP {e.response.status_code}: {url}")
        return False
    except Exception as e:
        print(f"  ✗ {filename} — {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("AgriN RAG — Downloading PDF corpus")
    print(f"Target directory: {os.path.abspath(PDF_DIR)}")
    print(f"Total PDFs to fetch: {len(ALL_PDFS)}")
    print("=" * 60)

    success, failed = [], []

    for section_name, pdfs in [
        ("NIPHM IPM Packages", NIPHM_PDFS),
        ("TNAU Crop Production Guides", TNAU_PDFS),
        ("PAU Package of Practices", PAU_PDFS),
        ("ICAR Seasonal Advisories", ICAR_PDFS),
    ]:
        print(f"\n── {section_name} ({len(pdfs)} files) ──")
        for filename, url in pdfs.items():
            ok = download_pdf(filename, url)
            (success if ok else failed).append(filename)
            time.sleep(0.5)  # polite delay

    print("\n" + "=" * 60)
    print(f"Downloaded: {len(success)}/{len(ALL_PDFS)}")
    if failed:
        print(f"\nFailed ({len(failed)}) — download these manually:")
        for f in failed:
            print(f"  • {f}: {ALL_PDFS[f]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
