#!/usr/bin/env bash
# JOA installer — Linux.
#   curl -fsSL https://raw.githubusercontent.com/TeamLider9141/JOA-AI-CODING-AGENT/main/install.sh | bash
#
# Env overrides (for local testing):
#   JOA_HOME     — install directory (default: ~/.joa)
#   JOA_REPO_URL — git URL to clone (default: the GitHub repo, can be file://...)
set -euo pipefail

REPO_URL="${JOA_REPO_URL:-https://github.com/TeamLider9141/JOA-AI-CODING-AGENT.git}"
INSTALL_DIR="${JOA_HOME:-$HOME/.joa}"
BIN_DIR="$HOME/.local/bin"

echo "
     ██╗ ██████╗  █████╗
     ██║██╔═══██╗██╔══██╗
     ██║██║   ██║███████║
██   ██║██║   ██║██╔══██║
╚█████╔╝╚██████╔╝██║  ██║
 ╚════╝  ╚═════╝ ╚═╝  ╚═╝
JOA — Lokal AI Coding Agent
"

# --- requirements -----------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "Xato: python3 topilmadi. O'rnating: https://python.org" >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "Xato: git topilmadi. O'rnating: https://git-scm.com" >&2
    exit 1
fi
if ! command -v ollama >/dev/null 2>&1; then
    echo "Ogohlantirish: ollama topilmadi. Lokal modellar (Ollama backend)"
    echo "uchun o'rnating: https://ollama.com/download"
    echo "(Gemini backend ollama'siz ham ishlaydi — --backend gemini)"
    echo
fi

# --- fetch --------------------------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "JOA allaqachon $INSTALL_DIR da bor — yangilanmoqda..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "JOA klonlanmoqda: $INSTALL_DIR"
    git clone --depth 1 --branch main "$REPO_URL" "$INSTALL_DIR"
fi

# --- venv + deps ----------------------------------------------------------
echo "Bog'liqliklar o'rnatilmoqda..."
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet -r "$INSTALL_DIR/assistant/requirements.txt"

# --- launcher symlink -----------------------------------------------------
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/bin/joa" "$BIN_DIR/joa"
chmod +x "$INSTALL_DIR/bin/joa"

echo
echo "✓ JOA o'rnatildi: $INSTALL_DIR"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        echo "$BIN_DIR PATH'da yo'q. Quyidagini shell konfiguratsiyangizga"
        echo "(~/.bashrc yoki ~/.zshrc) qo'shing:"
        echo
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        echo
        ;;
esac

echo "Keyingi qadam — Ollama modeli tortib oling (eng yengil/tezkori):"
echo
echo "  ollama pull qwen2.5-coder:0.5b   # eng tez, kichik CPU uchun"
echo "  ollama pull qwen2.5-coder:1.5b   # tezlik/sifat balansi"
echo "  ollama pull qwen2.5-coder:7b     # eng sifatli, sekinroq"
echo
echo "Keyin yangi terminal oching (yoki shellni qayta yuklang) va:"
echo
echo "  joa"
echo
