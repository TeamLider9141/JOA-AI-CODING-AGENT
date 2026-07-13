# Joa — Lokal AI Coding Agent

```
     ██╗ ██████╗  █████╗
     ██║██╔═══██╗██╔══██╗
     ██║██║   ██║███████║
██   ██║██║   ██║██╔══██║
╚█████╔╝╚██████╔╝██║  ██║
 ╚════╝  ╚═════╝ ╚═╝  ╚═╝
JOA — Lokal AI Coding Agent
```

Terminalda bu banner har doim **ko'k** rangda chiqadi — o'rnatishda
(`install.sh`) va `joa` REPL har ochilganida.

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

## Tezkor o'rnatish (Linux)

```
curl -fsSL https://raw.githubusercontent.com/TeamLider9141/JOA-AI-CODING-AGENT/main/install.sh | bash
ollama pull qwen2.5-coder:0.5b     # eng tez/yengil; yoki :1.5b / :3b / :7b
joa
```

`joa` yangi papkada birinchi marta ochilganda Claude Code'dagidek
workspace-trust ekrani chiqadi — strelka tugmalari bilan (↑/↓, Enter)
"Ha"/"Yo'q" tanlaysiz, keyin o'sha papka uchun qayta so'ramaydi. Papka
hali indekslanmagan bo'lsa, shu yerda hoziroq indekslashni ham taklif
qiladi — xuddi shu arrow-key menyu bilan (`~` kabi katta/aralash papkalar
uchun emas — aniq loyiha papkasi uchun mo'ljallangan).

Indekslash ikki bosqichda ishlaydi: **leksik (BM25) qism darhol** quriladi
(taxminan 0.1s — REPL shu zahoti ishlatishga tayyor), **semantik (vektor)
qism esa fonda**, sizga xalaqit bermay. Fon tugagach
`✓ Semantik qidiruv ham tayyor.` deb xabar chiqadi — o'shangacha qidiruv
faqat leksik (aniq so'z) ishlaydi. Fayllar o'zgarmagan bo'lsa, keyingi
`joa` ochilishlarida semantik qism qayta qurilmaydi (avtomatik aniqlanadi).

## Setup (qo'lda, loyihani o'zi ustida ishlash uchun)

```
python3 -m venv .venv
.venv/bin/pip install -r assistant/requirements.txt
ollama pull qwen2.5-coder:0.5b     # eng tez/yengil; yoki :1.5b / :3b / :7b
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
(masalan `qwen2.5-coder:1.5b`/`3b`/`7b`) yoki Gemini orasida almashtirish
mumkin — sessiyani qayta ishga tushirmasdan. Tanlov raqam kiritish orqali
emas, Claude Code'dagidek **strelka tugmalari** bilan: ↑/↓ ro'yxat bo'ylab
yuradi, Enter tanlaydi, Esc/Ctrl-C bekor qiladi. Joriy model boshlanishda
avtomatik belgilangan bo'ladi.

Ro'yxat rangli (Ollama modellari — moviy, gemini — pushti, joriy model —
yashil va `(joriy)` belgisi bilan). Har javobdan keyin qaysi model
javob berganini ko'rish uchun footer'da ham model nomi chiqadi:
`(2.3s · qwen2.5-coder:0.5b)`.

Oddiy savollar (masalan "bu funksiya nima qiladi?") endi agent
protokolisiz, bitta streaming chaqiruv bilan javob oladi — javob token
oqib chiqadi. Fayl/buyruq talab qiladigan topshiriqlar avvalgidek to'liq
agent orqali bajariladi (model o'zi ajratadi).

Boshqa slash-buyruqlar: `/` yoki `/help` — buyruqlar ro'yxati, `/clear` —
suhbat kontekstini tozalash (tarix 0 dan boshlanadi). `/` bilan boshlangan
kiritishlar hech qachon LLM'ga yuborilmaydi. Terminalda `/` yozganingizda
buyruqlar ro'yxati jonli dropdown sifatida chiqadi (yozgan sari
filtrlanadi, Tab/strelka bilan tanlanadi).

`!buyruq` — shell buyrug'ini LLM'siz to'g'ridan-to'g'ri bajaradi, chiqish
**jonli** oqadi (progress-barlar ham to'g'ri ko'rinadi, masalan
`!ollama pull qwen2.5-coder:0.5b`). Timeout yo'q — Ctrl-C bilan
to'xtatasiz. Agentning o'z `run_cmd` vositasi (masalan "buyruq bajar" deb
so'raganingizda) ham endi natijani kutmasdan jonli ko'rsatadi.

**Ctrl-C** — joriy amalni (model javob yozayotgani, agent ishlayotgani,
`!buyruq` bajarilayotgani) to'xtatadi, `joa`dan chiqarmaydi. `exit`/Ctrl-D
esa butun sessiyani yopadi.

## Testlar

```
.venv/bin/pytest
.venv/bin/python -m assistant.eval.run_eval --repo <repo-path>
```

Batafsil: [assistant/README.md](assistant/README.md)
