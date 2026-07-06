from pathlib import Path

from tree_sitter_language_pack import get_parser

from assistant.indexer.models import Chunk

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

FUNC_NODES = {"function_definition", "function_declaration", "method_definition"}
CLASS_NODES = {"class_definition", "class_declaration"}
WRAPPER_NODES = {"decorated_definition", "export_statement"}

TEXT_WINDOW = 80   # lines per fallback chunk
TEXT_OVERLAP = 15  # lines shared between consecutive fallback chunks


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    rel = str(path.relative_to(root))
    try:
        source = path.read_text(errors="ignore")
    except OSError:
        return []
    if not source.strip():
        return []

    lang = LANGUAGES.get(path.suffix.lower())
    if lang is None:
        return _chunk_text(rel, source)

    # This tree-sitter binding's parse() takes str, but node offsets
    # (start_byte/end_byte) are still UTF-8 byte offsets — so text
    # extraction slices the byte-encoded source, not the str.
    tree = get_parser(lang).parse(source)
    src_bytes = source.encode()
    chunks: list[Chunk] = []
    _collect(tree.root_node(), src_bytes, rel, chunks)
    return chunks or _chunk_text(rel, source)


def _children(node):
    return [node.child(i) for i in range(node.child_count())]


def _collect(node, src: bytes, rel: str, chunks: list[Chunk]) -> None:
    for child in _children(node):
        kind = child.kind()
        if kind in WRAPPER_NODES:
            _collect(child, src, rel, chunks)
        elif kind in CLASS_NODES:
            _collect_class(child, src, rel, chunks)
        elif kind in FUNC_NODES:
            chunks.append(
                _make_chunk(child, src, rel, _name(child, src), "function"))


def _collect_class(class_node, src: bytes, rel: str,
                   chunks: list[Chunk]) -> None:
    class_name = _name(class_node, src)
    header = _text(class_node, src).split("\n", 1)[0]
    body = class_node.child_by_field_name("body")

    methods = []
    for child in (_children(body) if body is not None else []):
        target = child
        if child.kind() in WRAPPER_NODES:
            target = next(
                (c for c in _children(child) if c.kind() in FUNC_NODES),
                child)
        if target.kind() in FUNC_NODES:
            methods.append(target)

    if not methods:
        chunks.append(_make_chunk(class_node, src, rel, class_name, "class"))
        return

    for m in methods:
        chunks.append(Chunk(
            path=rel,
            symbol=f"{class_name}.{_name(m, src)}",
            kind="method",
            start_line=m.start_position().row + 1,
            end_line=m.end_position().row + 1,
            text=f"{header}\n{_text(m, src)}",
        ))


def _make_chunk(node, src: bytes, rel: str, symbol: str, kind: str) -> Chunk:
    return Chunk(
        path=rel,
        symbol=symbol,
        kind=kind,
        start_line=node.start_position().row + 1,
        end_line=node.end_position().row + 1,
        text=_text(node, src),
    )


def _name(node, src: bytes) -> str:
    name_node = node.child_by_field_name("name")
    return _text(name_node, src) if name_node is not None else "anonymous"


def _text(node, src: bytes) -> str:
    return src[node.start_byte():node.end_byte()].decode(errors="ignore")


def _chunk_text(rel: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    step = TEXT_WINDOW - TEXT_OVERLAP
    for start in range(0, len(lines), step):
        window = lines[start:start + TEXT_WINDOW]
        if not window:
            break
        chunks.append(Chunk(
            path=rel,
            symbol=f"lines-{start + 1}",
            kind="text",
            start_line=start + 1,
            end_line=start + len(window),
            text="\n".join(window),
        ))
        if start + TEXT_WINDOW >= len(lines):
            break
    return chunks
