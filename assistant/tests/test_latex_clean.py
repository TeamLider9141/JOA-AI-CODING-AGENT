from assistant.latex_clean import clean_latex


def test_plain_text_without_latex_unchanged():
    assert clean_latex("just a normal sentence.") == "just a normal sentence."


def test_strips_paren_delimiters():
    assert clean_latex(r"\(x + 1\)") == "x + 1"


def test_strips_bracket_delimiters():
    assert clean_latex(r"\[x + 1\]") == "x + 1"


def test_strips_dollar_delimiters():
    assert clean_latex(r"$x + 1$") == "x + 1"
    assert clean_latex(r"$$x + 1$$") == "x + 1"


def test_strips_left_right_modifiers():
    assert clean_latex(r"\left(x + 1\right)") == "(x + 1)"


def test_greek_letters():
    assert clean_latex(r"\alpha + \beta = \gamma") == "α + β = γ"
    assert clean_latex(r"\Delta x") == "Δ x"


def test_common_symbols():
    assert clean_latex(r"\int_0^1") == "∫₀¹"
    assert clean_latex(r"\sum") == "∑"
    assert clean_latex(r"\infty") == "∞"
    assert clean_latex(r"a \leq b \geq c \neq d") == "a ≤ b ≥ c ≠ d"
    assert clean_latex(r"\sqrt{x}") == "√{x}"


def test_frac_becomes_slash():
    assert clean_latex(r"\frac{1}{2}") == "1/2"
    assert clean_latex(r"\frac{a+b}{c}") == "a+b/c"


def test_superscript_digit():
    assert clean_latex("x^2") == "x²"


def test_superscript_braced_multi_digit():
    assert clean_latex("x^{10}") == "x¹⁰"


def test_subscript_digit():
    assert clean_latex("x_1") == "x₁"


def test_subscript_braced():
    assert clean_latex("x_{12}") == "x₁₂"


def test_unrecognized_command_left_untouched():
    assert clean_latex(r"\text{hello}") == r"\text{hello}"


def test_unmappable_superscript_falls_back_unchanged():
    assert clean_latex("x^{q}") == "x^{q}"


def test_unmappable_subscript_falls_back_unchanged():
    assert clean_latex("x_{q}") == "x_{q}"
