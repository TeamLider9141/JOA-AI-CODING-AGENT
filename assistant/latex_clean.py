import re

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}

_SYMBOLS = {
    "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
    "sum": "∑", "prod": "∏", "infty": "∞", "partial": "∂", "nabla": "∇",
    "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈", "equiv": "≡",
    "pm": "±", "mp": "∓", "times": "×", "cdot": "·", "div": "÷",
    "to": "→", "rightarrow": "→", "leftarrow": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
    "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃",
    "emptyset": "∅", "therefore": "∴", "because": "∵",
    "sqrt": "√",
}

_SUPERSCRIPT = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼",
    "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ",
}

_SUBSCRIPT = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌",
    "(": "₍", ")": "₎", "a": "ₐ", "e": "ₑ", "o": "ₒ", "x": "ₓ",
}

_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SUP_RE = re.compile(r"\^\{([^{}]*)\}|\^(\w)")
_SUB_RE = re.compile(r"_\{([^{}]*)\}|_(\w)")
_DOLLAR_RE = re.compile(r"\${1,2}")


def _replace_command(match: re.Match) -> str:
    name = match.group(1)
    return _GREEK.get(name) or _SYMBOLS.get(name) or match.group(0)


def _to_script(chars: str, table: dict) -> str | None:
    out = []
    for c in chars:
        if c not in table:
            return None
        out.append(table[c])
    return "".join(out)


def _replace_script(match: re.Match, table: dict) -> str:
    inner = match.group(1) if match.group(1) is not None else match.group(2)
    converted = _to_script(inner, table)
    return converted if converted is not None else match.group(0)


def clean_latex(text: str) -> str:
    """Best-effort conversion of common LaTeX markup to readable Unicode
    for terminal display (small local models often answer math questions
    in raw LaTeX, which renders as unreadable escape-sequence soup in a
    plain terminal). Doesn't cover the full LaTeX grammar — unrecognized
    commands and superscript/subscript characters with no Unicode
    equivalent are left untouched rather than risk corrupting the text."""
    text = _FRAC_RE.sub(r"\1/\2", text)
    text = _COMMAND_RE.sub(_replace_command, text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    for delim in (r"\(", r"\)", r"\[", r"\]"):
        text = text.replace(delim, "")
    text = _DOLLAR_RE.sub("", text)
    text = _SUP_RE.sub(lambda m: _replace_script(m, _SUPERSCRIPT), text)
    text = _SUB_RE.sub(lambda m: _replace_script(m, _SUBSCRIPT), text)
    return text
