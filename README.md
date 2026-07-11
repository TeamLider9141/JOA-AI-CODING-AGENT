# Joa — Lokal AI Coding Agent

CPU-only, offline-first kod-yordamchi. Default holatda Ollama orqali lokal
LLM ishlatadi — cloud API'ga bog'liq emas, kod tashqariga chiqmaydi. Xohlasa,
tezlik uchun `--backend gemini` bilan Google Gemini'ga ham o'tish mumkin
(ixtiyoriy, cloud API key talab qiladi).

## Nima qiladi

- **`joa`** — interaktiv REPL CLI, ko'p bosqichli suhbatni saqlaydi
- **Tool-use agent** — fayl o'qish/yozish, buyruq bajarish, kod qidirish
- **Gibrid qidiruv** — tree-sitter AST chunking + BM25 (leksik) + Qdrant
  (vektor semantik) qidiruv, RRF bilan birlashtirilgan
- **LLM reranker** — qidiruv natijalarini qayta tartiblash (ishlab
  chiqilmoqda)
- **Eval harness** — gold-standard savol-javoblar bilan avtomatik sifat
  o'lchash

## Texnologiyalar

Python, Typer (CLI), Ollama (qwen2.5-coder, nomic-embed-text), Gemini API
(ixtiyoriy ikkinchi chat backend), Qdrant, rank_bm25, tree-sitter, pytest
(TDD, 120+ test)

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r assistant/requirements.txt
ollama pull qwen2.5-coder:7b       # yoki :3b / :1.5b — tezroq, CPU uchun yengilroq
ollama pull nomic-embed-text
```

Gemini backend'ni ishlatmoqchi bo'lsangiz (ixtiyoriy):

```
cp .env.example .env
# .env faylga GEMINI_API_KEY qo'shing (https://aistudio.google.com/apikey)
```

## Ishlatish

```
.venv/bin/python -m assistant.cli index <repo-path>
.venv/bin/python -m assistant.cli search "query" --repo <repo-path>
.venv/bin/python -m assistant.cli ask "question" --repo <repo-path>
.venv/bin/python -m assistant.cli ask "question" --repo <repo-path> --backend gemini
bin/joa                                    # interaktiv REPL (Ollama)
bin/joa --backend gemini                   # interaktiv REPL (Gemini)
```

`--backend` — `ask`, `agent`, `repl` buyruqlarida ishlaydi (`index`/`search`da
yo'q, chunki ular faqat embedding ishlatadi — embedding doim Ollama'da
qoladi, backend tanlovidan qat'i nazar). Default: `ollama`.

REPL sessiyasi ichida `/joamodel` yozib, o'rnatilgan Ollama modellari
(masalan `qwen2.5-coder:1.5b`/`3b`/`7b`) yoki Gemini orasida raqam bilan
almashtirish mumkin — sessiyani qayta ishga tushirmasdan:

```
joa> /joamodel
1. qwen2.5-coder:1.5b
2. qwen2.5-coder:7b
3. gemini
Raqamni tanlang:
joa> 1
✓ Model: qwen2.5-coder:1.5b
```

## Testlar

```
.venv/bin/pytest
.venv/bin/python -m assistant.eval.run_eval --repo <repo-path>
```

Batafsil: [assistant/README.md](assistant/README.md)
