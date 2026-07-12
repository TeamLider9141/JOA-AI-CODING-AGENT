import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from `path` into os.environ (never overriding
    a variable that's already set in the real environment)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).parent.parent / ".env")

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 4096            # CPU-only: keep modest, tune later
REQUEST_TIMEOUT = 300.0   # seconds; CPU inference is slow
KEEP_ALIVE = "30m"        # keep models in RAM between requests ("-1" = forever);
                          # avoids a full CPU model reload after Ollama's 5m default

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com"

# --- Retrieval ---
VECTOR_TOP_K = 40
BM25_TOP_K = 40
RRF_K = 60
FINAL_TOP_K = 10

# --- Paths ---
DATA_DIR = Path(__file__).parent / ".data"

# --- Agent ---
RUN_CMD_TIMEOUT = 120  # seconds; generous enough for git push on slow links
