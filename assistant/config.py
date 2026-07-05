from pathlib import Path

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 4096            # CPU-only: keep modest, tune later
REQUEST_TIMEOUT = 300.0   # seconds; CPU inference is slow

# --- Retrieval ---
VECTOR_TOP_K = 40
BM25_TOP_K = 40
RRF_K = 60
FINAL_TOP_K = 10

# --- Paths ---
DATA_DIR = Path(__file__).parent / ".data"
