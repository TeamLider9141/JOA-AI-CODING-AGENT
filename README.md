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

[![Klonlar soni](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/TeamLider9141/JOA-AI-CODING-AGENT/main/.github/badges/clone-count.json)](https://github.com/TeamLider9141/JOA-AI-CODING-AGENT)

<!-- CLONE_CHART:START -->
![Klonlar grafigi](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%2207-01%22%2C%2207-02%22%2C%2207-03%22%2C%2207-04%22%2C%2207-05%22%2C%2207-06%22%2C%2207-07%22%2C%2207-08%22%2C%2207-09%22%2C%2207-10%22%2C%2207-11%22%2C%2207-12%22%2C%2207-13%22%2C%2207-14%22%2C%2207-15%22%2C%2207-16%22%2C%2207-17%22%2C%2207-18%22%2C%2207-19%22%2C%2207-20%22%2C%2207-21%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Umumiy%20klonlar%22%2C%22data%22%3A%5B0%2C0%2C0%2C0%2C0%2C0%2C0%2C12%2C13%2C13%2C82%2C136%2C183%2C185%2C221%2C224%2C230%2C238%2C243%2C251%2C253%5D%2C%22borderColor%22%3A%22%232563eb%22%2C%22backgroundColor%22%3A%22rgba%2837%2C99%2C235%2C0.15%29%22%2C%22fill%22%3Atrue%2C%22tension%22%3A0.3%2C%22pointRadius%22%3A0%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22legend%22%3A%7B%22display%22%3Afalse%7D%2C%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Repo%20klonlari%20%28kunlik%20yig%27indi%29%22%7D%7D%2C%22scales%22%3A%7B%22y%22%3A%7B%22beginAtZero%22%3Atrue%7D%7D%7D%7D&width=700&height=320&backgroundColor=white)
<!-- CLONE_CHART:END -->

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
(TDD, 240+ test)

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
protokolisiz, bitta chaqiruv bilan javob oladi (to'liq javob tayyor
bo'lgach chiqadi — LaTeX formulalar bo'lsa, avval Unicode'ga tozalanadi:
`\alpha` → α, `\frac{1}{2}` → 1/2 va h.k). Fayl/buyruq talab qiladigan
topshiriqlar avvalgidek to'liq agent orqali bajariladi (model o'zi
ajratadi).

Kichik modellar (`0.5b`/`1.5b`) kod savollarida yaxshi, lekin umumiy
bilim/matematik hisob-kitobda ishonchsiz bo'lishi mumkin (xato javobni
ham ishonch bilan aytib yuborishi mumkin). Murakkab/nostandart savollar
uchun `/joamodel` bilan kattaroq modelga (`7b`) yoki Gemini'ga o'tish
tavsiya qilinadi.

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

## Kuzatuvda: Bonsai 27B

[PrismML'ning Bonsai 27B](https://docs.prismml.com/models/bonsai-27b)i
(1-bit/ternary, 27B, Apache 2.0) diqqatga sazovor — 1-bit versiyasi
faqat ~3.5GB, ammo Ollama hali uning Q1_0 GGUF formatini yuklay olmaydi
(Ollama'ning `ggml` build'ida bu tur yo'q). Ollama qo'llab-quvvatlashni
qo'shgach, Joa'da qo'shimcha kod kerak bo'lmaydi — `/joamodel`
`ollama pull`langan istalgan modelni avtomatik ko'radi.

## Testlar

```
.venv/bin/pytest
.venv/bin/python -m assistant.eval.run_eval --repo <repo-path>
```

Batafsil: [assistant/README.md](assistant/README.md)
