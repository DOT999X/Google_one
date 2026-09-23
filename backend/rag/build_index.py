"""
AgriN RAG — Build Vector Index
Run this once after downloading PDFs to create the ChromaDB index.

Usage:
    python -m rag.build_index              # build (skip if exists)
    python -m rag.build_index --rebuild     # force rebuild
"""

import sys
import logging
from .ingest import ingest_all_pdfs
from .embeddings import init_embedding_model
from .store import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("agrin.rag.build")
from dotenv import load_dotenv
load_dotenv()

def main():
    force = "--rebuild" in sys.argv

    logger.info("Step 1: Initializing embedding model...")
    init_embedding_model()

    logger.info("Step 2: Ingesting PDFs...")
    chunks = ingest_all_pdfs()
    if not chunks:
        logger.error("No chunks produced — check that PDFs exist in rag/data/pdfs/")
        sys.exit(1)

    logger.info(f"Step 3: Building vector index ({len(chunks)} chunks)...")
    build_index(chunks, force_rebuild=force)

    logger.info("Done! RAG index is ready.")


if __name__ == "__main__":
    main()
