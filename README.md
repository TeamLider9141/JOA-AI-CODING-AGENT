# Joa — Lokal AI Coding Agent

CPU-only, to'liq offline ishlaydigan kod-yordamchi. Ollama orqali lokal LLM
ishlatadi — cloud API'ga bog'liq emas, kod tashqariga chiqmaydi.

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

Python, Typer (CLI), Ollama (qwen2.5-coder:7b, nomic-embed-text),
Qdrant, rank_bm25, tree-sitter, pytest (TDD, 96+ test)

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r assistant/requirements.txt
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

## Ishlatish

```
.venv/bin/python -m assistant.cli index <repo-path>
.venv/bin/python -m assistant.cli search "query" --repo <repo-path>
.venv/bin/python -m assistant.cli ask "question" --repo <repo-path>
bin/joa                                    # interaktiv REPL
```

## Testlar

```
.venv/bin/pytest
.venv/bin/python -m assistant.eval.run_eval --repo <repo-path>
```

Batafsil: [assistant/README.md](assistant/README.md)
