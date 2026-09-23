"""
AgriN RAG — PDF Ingestion & Structural Chunking

Strategy: Option B — parse PDF heading hierarchy, chunk by section,
tag each chunk with crop/topic/disease metadata for filtered retrieval.
"""

import os
import re
import json
import hashlib
import pymupdf as fitz  # PyMuPDF (new import name, same API)
import logging

logger = logging.getLogger("agrin.rag.ingest")

CHUNK_DIR = os.path.join(os.path.dirname(__file__), "data", "chunks")
PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
os.makedirs(CHUNK_DIR, exist_ok=True)

# ── Crop name mapping: filename patterns → canonical crop names ──────────
FILENAME_TO_CROP = {
    "tomato": "Tomato", "rice": "Rice", "potato": "Potato",
    "maize": "Corn (maize)", "grape": "Grape", "apple": "Apple",
    "cotton": "Cotton", "soybean": "Soybean", "citrus": "Orange",
    "chilli": "Pepper, bell", "strawberry": "Strawberry", "peach": "Peach",
    "cherry": "Cherry (including sour)", "sugarcane": "Sugarcane",
    "wheat": "Wheat", "groundnut": "Groundnut", "mustard": "Mustard",
    "lentil": "Lentil", "chickpea": "Chickpea", "pigeonpea": "Pigeonpea",
    "banana": "Banana", "mango": "Mango", "onion": "Onion",
    "brinjal": "Brinjal", "okra": "Okra", "cabbage": "Cabbage",
    "cauliflower": "Cauliflower",
}

# ── Topic classification keywords ────────────────────────────────────────
TOPIC_KEYWORDS = {
    "disease_management": [
        "disease", "blight", "wilt", "rot", "rust", "mildew", "mosaic",
        "virus", "fungal", "bacterial", "fungicide", "pathogen", "scab",
        "canker", "leaf spot", "smut", "anthracnose", "damping off",
    ],
    "pest_management": [
        "pest", "insect", "borer", "aphid", "whitefly", "mite", "thrip",
        "caterpillar", "weevil", "hopper", "moth", "larva", "nematode",
        "insecticide", "pesticide", "spray", "trap", "pheromone",
    ],
    "fertilizer": [
        "fertilizer", "fertiliser", "manure", "nitrogen", "phosphorus",
        "potassium", "npk", "urea", "dap", "mop", "fym", "compost",
        "nutrient management", "micronutrient", "zinc", "boron",
        "nutrient deficiency", "dose", "basal", "top dressing",
    ],
    "cultural_practice": [
        "sowing", "planting", "transplanting", "seed rate", "spacing",
        "irrigation", "weeding", "mulching", "intercropping", "rotation",
        "land preparation", "nursery", "field preparation", "pruning",
        "thinning", "earthing up", "staking",
    ],
    "variety": [
        "variety", "varieties", "cultivar", "hybrid", "seed",
        "resistant variety", "tolerant", "duration", "maturity",
        "high yielding", "recommended varieties",
    ],
    "harvest": [
        "harvest", "post-harvest", "storage", "drying", "threshing",
        "grading", "marketing", "yield", "maturity index",
    ],
}

# ── Known disease names (from PlantVillage label map) ────────────────────
KNOWN_DISEASES = [
    "apple scab", "black rot", "cedar apple rust", "powdery mildew",
    "cercospora leaf spot", "gray leaf spot", "common rust",
    "northern leaf blight", "esca", "black measles", "leaf blight",
    "isariopsis", "citrus greening", "huanglongbing", "bacterial spot",
    "early blight", "late blight", "leaf mold", "septoria leaf spot",
    "spider mite", "target spot", "yellow leaf curl", "mosaic virus",
    "leaf scorch", "downy mildew", "anthracnose", "fusarium wilt",
    "bacterial wilt", "root rot", "stem rot", "damping off",
]


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF, preserving page structure."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num + 1, "text": text.strip()})
    doc.close()
    return pages


def detect_crop_from_filename(filename: str) -> str:
    """Infer crop from PDF filename."""
    name_lower = filename.lower()
    for pattern, crop in FILENAME_TO_CROP.items():
        if pattern in name_lower:
            return crop
    return "general"


def classify_topic(text: str) -> str:
    """Classify chunk topic based on keyword density."""
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = score

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def extract_diseases(text: str) -> list[str]:
    """Find disease names mentioned in the text."""
    text_lower = text.lower()
    found = []
    for disease in KNOWN_DISEASES:
        if disease in text_lower:
            found.append(disease)
    return found



# ── NIPHM template anchor labels ──────────────────────────────────────────
# These recurring sub-section labels are the true structural backbone of
# NIPHM IPM packages (confirmed by inspecting raw extracted text) — far
# more reliable than generic heading heuristics for this document family.
CONTROL_LABELS = {
    "damage symptoms", "survival and spread", "favourable conditions",
    "cultural control", "chemical control", "biological control",
    "mechanical control", "physical control", "botanical control",
    "nutrient management", "weed management", "management",
    "biology", "egg", "larva", "pupa", "adult", "nymph",
    "identification", "distribution", "symptoms", "nature of damage",
    "economic threshold level", "etl", "monitoring", "prevention",
    "host range", "life cycle", "insecticide resistance management",
}


def detect_section_headers(text: str) -> list[tuple[int, str]]:
    """
    Detect section headers in text.

    Anchors on three signals, in order of reliability:
    1. NIPHM template labels (Cultural control, Chemical control, Biology, etc.)
       — the actual recurring structure of these government IPM documents.
       When found, the preceding short candidate line (pest/disease name)
       is prefixed so the chunk carries its subject, e.g.
       "Early blight — Chemical control".
    2. Numbered pest/disease headers like "4) Early blight:" or
       "1) Serpentine leaf miner:" — requires a colon or 2+ words to avoid
       matching numbered list items like "3. Pupa" from biology tables.
    3. ALL CAPS lines — top-level section headers (I. PESTS, VII. ...).
    """
    headers = []
    lines = text.split("\n")
    pending_name = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            continue

        label_check = stripped.rstrip(":").strip().lower()

        # Signal 1: known NIPHM template label
        if label_check in CONTROL_LABELS:
            header_text = f"{pending_name} — {stripped.rstrip(':')}" if pending_name else stripped.rstrip(":")
            headers.append((i, header_text))
            continue

        # Signal 2: numbered pest/disease header — "4) Early blight:" or "1) Serpentine leaf miner:"
        m = re.match(r'^\d+[\.\)]\s+([A-Za-z][A-Za-z\s,\-]{2,70})', stripped)
        if m:
            title = m.group(1).strip().rstrip(":").strip()
            has_colon = stripped.rstrip().endswith(":")
            multi_word = len(title.split()) >= 2
            if has_colon or multi_word:
                headers.append((i, title))
                pending_name = title
                continue

        # Signal 3: ALL CAPS lines — top-level sections (I. PESTS, VII. Description...)
        if stripped.isupper() and 5 < len(stripped) < 80 and not stripped.startswith("•"):
            headers.append((i, stripped))
            pending_name = None
            continue

        # Track a candidate pest/disease name line — short, capitalized,
        # not ending in a period — so the NEXT control label can inherit it.
        # This does not itself become a header/split point.
        if (len(stripped.split()) <= 8 and len(stripped) < 60
                and not stripped.endswith(".") and stripped[:1].isupper()
                and not stripped.endswith(",")):
            pending_name = stripped

    return headers


def chunk_by_sections(pages: list[dict], max_chunk_tokens: int = 1000,
                      min_chunk_tokens: int = 50) -> list[dict]:
    """
    Structural chunking: split on detected headers, respecting token limits.
    Falls back to paragraph-based splitting for headerless pages.
    """
    full_text = "\n\n".join(p["text"] for p in pages)
    lines = full_text.split("\n")
    headers = detect_section_headers(full_text)

    chunks = []

    if len(headers) >= 3:
        # Section-based chunking
        for idx, (line_num, header) in enumerate(headers):
            # Get text until next header
            next_line = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
            section_lines = lines[line_num:next_line]
            section_text = "\n".join(section_lines).strip()

            if not section_text:
                continue

            # Approximate token count (rough: 1 token ≈ 4 chars)
            approx_tokens = len(section_text) / 4

            if approx_tokens < min_chunk_tokens:
                continue  # too small, will be captured by next section

            if approx_tokens <= max_chunk_tokens:
                chunks.append({"header": header, "text": section_text})
            else:
                # Split large sections at paragraph boundaries
                paragraphs = re.split(r'\n\s*\n', section_text)
                current_chunk = []
                current_len = 0
                for para in paragraphs:
                    para_tokens = len(para) / 4
                    if current_len + para_tokens > max_chunk_tokens and current_chunk:
                        chunks.append({
                            "header": header,
                            "text": "\n\n".join(current_chunk),
                        })
                        current_chunk = [para]
                        current_len = para_tokens
                    else:
                        current_chunk.append(para)
                        current_len += para_tokens
                if current_chunk:
                    chunks.append({
                        "header": header,
                        "text": "\n\n".join(current_chunk),
                    })
    else:
        # Fallback: paragraph-based chunking with overlap
        paragraphs = re.split(r'\n\s*\n', full_text)
        current_chunk = []
        current_len = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = len(para) / 4
            if current_len + para_tokens > max_chunk_tokens and current_chunk:
                chunks.append({
                    "header": "Section",
                    "text": "\n\n".join(current_chunk),
                })
                # Keep last paragraph as overlap
                current_chunk = [current_chunk[-1], para] if current_chunk else [para]
                current_len = sum(len(c) / 4 for c in current_chunk)
            else:
                current_chunk.append(para)
                current_len += para_tokens
        if current_chunk:
            chunks.append({
                "header": "Section",
                "text": "\n\n".join(current_chunk),
            })

    return chunks


def process_pdf(pdf_path: str, source_type: str = "NIPHM_IPM") -> list[dict]:
    """Full pipeline: extract → chunk → tag metadata."""
    filename = os.path.basename(pdf_path)
    crop = detect_crop_from_filename(filename)

    logger.info(f"Processing {filename} (crop={crop})")

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        logger.warning(f"No text extracted from {filename}")
        return []

    raw_chunks = chunk_by_sections(pages)
    tagged_chunks = []

    for i, chunk in enumerate(raw_chunks):
        header = chunk["header"]
        # Prepend header/subject to the chunk text — without this, a chunk's
        # pure chemical-dosage content (e.g. under "Early blight — Chemical
        # control") never mentions "early blight" anywhere in its own text,
        # so both semantic search and disease-name filtering silently miss it.
        text = f"{header}\n\n{chunk['text']}" if header and header != "Section" else chunk["text"]
        topic = classify_topic(text)
        diseases = extract_diseases(text)
        chunk_id = hashlib.md5(f"{filename}:{i}:{chunk['header'][:30]}".encode()).hexdigest()[:12]

        tagged_chunks.append({
            "id": f"{filename.replace('.pdf', '')}_{chunk_id}",
            "text": text,
            "header": chunk["header"],
            "metadata": {
                "crop": crop,
                "topic": topic,
                "diseases": diseases,
                "source": source_type,
                "source_file": filename,
                "source_url": f"niphm.gov.in/IPMPackages/{filename.replace('ipm_', '').replace('.pdf', '').title()}.pdf",
                "chunk_index": i,
            },
        })

    logger.info(f"  → {len(tagged_chunks)} chunks from {filename}")
    return tagged_chunks


def ingest_all_pdfs() -> list[dict]:
    """Process all PDFs in data/pdfs/ directory."""
    all_chunks = []

    if not os.path.exists(PDF_DIR):
        logger.error(f"PDF directory not found: {PDF_DIR}")
        return []

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDFs found in {PDF_DIR}")
        return []

    logger.info(f"Ingesting {len(pdf_files)} PDFs from {PDF_DIR}")

    for filename in sorted(pdf_files):
        pdf_path = os.path.join(PDF_DIR, filename)

        # Detect source type from filename
        if filename.startswith("ipm_"):
            source_type = "NIPHM_IPM"
        elif filename.startswith("tnau_"):
            source_type = "TNAU_CPG"
        elif filename.startswith("pau_"):
            source_type = "PAU_POP"
        elif filename.startswith("icar_"):
            source_type = "ICAR_Advisory"
        else:
            source_type = "Other"

        try:
            chunks = process_pdf(pdf_path, source_type)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    # Save chunks to JSON for debugging
    chunks_file = os.path.join(CHUNK_DIR, "all_chunks.json")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved {len(all_chunks)} chunks to {chunks_file}")

    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = ingest_all_pdfs()
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"Sample chunk metadata: {json.dumps(chunks[0]['metadata'], indent=2)}")
        print(f"Sample text preview: {chunks[0]['text'][:200]}...")
