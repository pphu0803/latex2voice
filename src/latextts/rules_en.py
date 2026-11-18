"""LaTeX -> English spoken-form rules. The core asset of the project.

Each rule maps a LaTeX pattern to canonical English reading.
Ambiguity policy: one canonical reading per construct (evaluation needs a unique answer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Reading:
    latex: str
    spoken: str


# Greek letters
GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "iota": "iota", "kappa": "kappa", "lambda": "lambda",
    "mu": "mu", "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho",
    "sigma": "sigma", "tau": "tau", "upsilon": "upsilon", "phi": "phi",
    "varphi": "phi", "chi": "kai", "psi": "psi", "omega": "omega",
    "Gamma": "capital gamma", "Delta": "capital delta", "Theta": "capital theta",
    "Lambda": "capital lambda", "Xi": "capital xi", "Pi": "capital pi",
    "Sigma": "capital sigma", "Phi": "capital phi", "Psi": "capital psi",
    "Omega": "capital omega",
}

# Binary operators
OPERATORS = {
    r"\times": " times ",
    r"\cdot": " times ",
    r"\div": " divided by ",
    r"\pm": " plus or minus ",
    r"\mp": " minus or plus ",
    r"\cup": " union ",
    r"\cap": " intersect ",
    r"\in": " is in ",
    r"\notin": " is not in ",
    r"\subset": " is a subset of ",
    r"\subseteq": " is a subset of or equal to ",
    r"\to": " to ",
    r"\rightarrow": " to ",
    r"\mapsto": " maps to ",
    r"\approx": " is approximately ",
    r"\neq": " is not equal to ",
    r"\ne": " is not equal to ",
    r"\leq": " is less than or equal to ",
    r"\le": " is less than or equal to ",
    r"\geq": " is greater than or equal to ",
    r"\ge": " is greater than or equal to ",
    r"\sim": " is similar to ",
    r"\propto": " is proportional to ",
    r"\equiv": " is identically equal to ",
    r"\wedge": " and ",
    r"\vee": " or ",
    r"\neg": " not ",
    r"\forall": " for all ",
    r"\exists": " there exists ",
    r"\Rightarrow": " implies ",
    r"\rightarrow": " to ",
    r"\leftrightarrow": " if and only if ",
    r"\Leftrightarrow": " if and only if ",
    r"\emptyset": " the empty set ",
    r"\infty": "infinity",
    r"\partial": "partial ",
    r"\nabla": "nabla ",
    r"\odot": " elementwise product ",
    r"\otimes": " tensor product ",
    r"\oplus": " direct sum ",
    r"\ell": "ell",
    r"\hbar": "h bar",
}

# ellipsis variants read as "and so on" (mainstream spoken convention)
_DOTS = {
    r"\dots": " and so on ", r"\ldots": " and so on ", r"\cdots": " and so on ",
    r"\vdots": " and so on ", r"\ddots": " and so on ",
}

# Functions read as their name
FUNCTIONS = {
    r"\sin": " sine", r"\cos": " cosine", r"\tan": " tangent",
    r"\cot": " cotangent", r"\sec": " secant", r"\csc": " cosecant",
    r"\log": " log", r"\ln": " natural log", r"\exp": " exp",
    r"\sinh": " hyperbolic sine", r"\cosh": " hyperbolic cosine",
    r"\tanh": " hyperbolic tangent", r"\arcsin": " arc sine",
    r"\arccos": " arc cosine", r"\arctan": " arc tangent",
    r"\min": " min", r"\max": " max", r"\det": " determinant of",
    r"\sup": " supremum of", r"\inf": " infimum of",
    r"\gcd": " the greatest common divisor of", r"\deg": " the degree of",
    r"\dim": " the dimension of", r"\arg": " the argument of",
    r"\limsup": " the limit superior of", r"\liminf": " the limit inferior of",
    r"\Re": " the real part of", r"\Im": " the imaginary part of",
    r"\Pr": " the probability of",
    r"\hom": " the hom set of", r"\ker": " the kernel of",
    r"\lg": " log base 2 of", r"\coth": " hyperbolic cotangent",
    r"\tr": " the trace of", r"\rank": " the rank of",
    r"\diag": " the diagonal of", r"\sign": " the sign of",
    r"\infty": " infinity",
}

# Common named operators without a backslash macro in amsmath, read verbatim.
# Source of authority for the canonical operator list: the amsmath package
# (\DeclareMathOperator set). \operatorname{...} and \mathrm{...} pass names
# through verbatim, so these only need word-level pronunciation fixes.
NAME_FIXES = {
    "softmax": "softmax",
    "sigmoid": "sigmoid",
    "relu": "R e L U",
    "gelu": "G E L U",
    "layernorm": "layer norm",
    "batchnorm": "batch norm",
    "concat": "concat",
}


def _spell_symbol(c: str) -> str:
    if c.isalpha():
        return c
    return {
        "+": " plus ", "-": " minus ", "*": " times ", "=": " equals ",
        "<": " less than ", ">": " greater than ", ",": ", ",
        "(": " , ", ")": " , ",  # grouping parens become pauses; function parens handled separately
        "[": " open bracket ", "]": " close bracket ",
        "{": " open brace ", "}": " close brace ",
        "|": " absolute value of ", "!": " factorial",
        "^": " to the power ", "_": " sub ",
        "/": " over ", ".": " point ", ":": " to ",
    }.get(c, " ")


def latex_to_spoken(latex: str) -> str:
    """Rule-based LaTeX -> English spoken form. Covers L1/L2 patterns.

    Returns text with extra spaces; caller normalizes whitespace.
    """
    s = latex.strip()
    s = re.sub(r"\s+", " ", s)

    # \left \right are invisible — but only when followed by a delimiter,
    # so \leftrightarrow / \rightarrow are left alone
    s = re.sub(r"\\left\s*(?=[(\[{|.\\,])", "", s)
    s = re.sub(r"\\right\s*(?=[)\]}|.\\,])", "", s)

    # \frac{a}{b} -> ( a over b ), recursively for nested braces
    _NESTED = r"\{((?:[^{}]|\{[^{}]*\})*)\}"

    def _frac(m: re.Match) -> str:
        num = latex_to_spoken(m.group(1))
        den = latex_to_spoken(m.group(2))
        return f" ( {num.strip()} over {den.strip()} ) "

    for _ in range(6):
        new = re.sub(r"\\(?:d)?frac" + _NESTED + _NESTED, _frac, s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\(?:d)?tfrac\{([^{}]*)\}\{([^{}]*)\}", r" \1 over \2 ", s)

    # \sqrt[n]{x} -> the n-th root of x ; \sqrt{x} -> the square root of x
    s = re.sub(r"\\sqrt\[(\w+)\]" + _NESTED, r" the \1 th root of \2 ", s)
    s = re.sub(r"\\sqrt" + _NESTED, r" the square root of \1 ", s)

    # integrals / sums / products / limits — must run BEFORE generic sub/superscript
    s = re.sub(r"\\int_\{?([\w.\\]+)\}?\^\{?([\w.\\]+)\}?", r" the integral from \1 to \2 of ", s)
    s = re.sub(r"\\int_\{?([\w.\\]+)\}?", r" the integral from \1 of ", s)
    s = re.sub(r"\\int\b", " the integral of ", s)
    s = re.sub(r"\\sum_\{(\w+ *= *[\w.]+)\}\^\{?([\w.]+)\}?", r" the sum from \1 to \2 of ", s)
    s = re.sub(r"\\sum_\{?([\w.]+)\}?\^\{?([\w.]+)\}?", r" the sum from \1 to \2 of ", s)
    s = re.sub(r"\\sum_\{?([\w.]+)\}?", r" the sum over \1 of ", s)
    s = re.sub(r"\\sum\b", " the sum of ", s)
    s = re.sub(r"\\prod_\{(\w+ *= *[\w.]+)\}\^\{?([\w.]+)\}?", r" the product from \1 to \2 of ", s)
    s = re.sub(r"\\prod\b", " the product of ", s)
    s = re.sub(r"\\lim_\{(\w+)\s*\\to\s*\\infty\}", r" the limit as \1 goes to infinity of ", s)
    s = re.sub(r"\\lim_\{([^{}]*)\}", r" the limit as \1 of ", s)
    s = re.sub(r"\\lim\b", " the limit of ", s)

    # \circ: degrees in superscript, composition between functions
    s = re.sub(r"\^\{?\\circ\}?", " degrees ", s)
    s = re.sub(r"(\w)\s*\\circ\s*(\w)", r" \1 composed with \2 ", s)

    # GROUPING: (expr)^{p} — parens change the order of operations, must be spoken.
    # "the quantity a plus b, squared". Run BEFORE generic superscript conversion.
    def _group_power(m: re.Match) -> str:
        inner = latex_to_spoken(m.group(1))
        expo = latex_to_spoken(m.group(2))
        return f" the quantity {inner.strip()}, {expo.strip()} "

    for _ in range(4):
        new = re.sub(r"\(([^()]+)\)\s*(\^" + _NESTED + r"|\^\w)", _group_power, s)
        if new == s:
            break
        s = new

    # superscripts: ^{...} or ^single
    s = re.sub(r"\^" + _NESTED, r" to the power \1 ", s)
    s = re.sub(r"\^(\w)", r" to the power \1 ", s)
    # the common squares/cubes read naturally
    s = re.sub(r"to the power 2\b", "squared", s)
    s = re.sub(r"to the power 3\b", "cubed", s)
    s = re.sub(r"to the power -1\b", "to the power minus one", s)
    # grouping naturalization: "the quantity a plus b, to the power 2" -> squared
    s = re.sub(r"the quantity ([^,]+), to the power 2\b", r"the quantity \1, squared", s)
    s = re.sub(r"the quantity ([^,]+), to the power 3\b", r"the quantity \1, cubed", s)
    s = re.sub(r"the quantity ([^,]+), to the power -1\b", r"the quantity \1, to the power minus one", s)

    # subscripts: _{...} or _single
    s = re.sub(r"_" + _NESTED, r" sub \1 ", s)
    s = re.sub(r"_(\w)", r" sub \1 ", s)

    # derivatives
    s = re.sub(r"\\frac\{d(\w+)\}\{d(\w+)\}", r" the derivative of \1 with respect to \2 ", s)
    s = re.sub(r"(\w)'\(", r"\1 prime of (", s)
    s = re.sub(r"(\w)'", r"\1 prime", s)

    # L3 structures: \text{...} keeps words verbatim
    s = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", s)
    # \operatorname{...} / \mathrm{...}: keep the name, fix pronunciation of known ones
    def _operator(m: re.Match) -> str:
        word = m.group(1).strip()
        return " " + NAME_FIXES.get(word.lower(), word) + " "
    s = re.sub(r"\\operatorname\{([^{}]*)\}", _operator, s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", _operator, s)

    # \|v\| or \left\| v \right\| -> the norm of v (matches \| ... \| pairs)
    s = re.sub(r"\\\|\s*(.*?)\s*\\\|", r" the norm of \1 ", s)
    # ^\top / ^T -> transpose
    s = re.sub(r"\^\{?\\top\}?", " transpose ", s)
    s = re.sub(r"\^\{?T\}?", " transpose ", s)

    # 2x2 pmatrix/bmatrix: & = column separator, \\ = row separator
    # mainstream reading: "the 2 by 2 matrix with first row a, b, and second row c, d"
    def _matrix(m: re.Match) -> str:
        body = m.group(1)
        rows = [r.strip() for r in body.split(r"\\")]
        rows = [r for r in rows if r]
        n, k = len(rows), len(rows[0].split("&"))
        parts = [f"the {n} by {k} matrix with"]
        for i, row in enumerate(rows):
            cells = [c.strip() for c in row.split("&")]
            word = ["first", "second", "third", "fourth"][i] if i < 4 else f"row {i+1}"
            conj = "and " if i == len(rows) - 1 else ""
            parts.append(f"{conj}{word} row " + ", ".join(cells))
        return " " + ", ".join(parts) + " "

    s = re.sub(
        r"\\begin\{(?:p|b|v)?matrix\}(.*?)\\end\{(?:p|b|v|V|B|P)?matrix\}",
        _matrix,
        s,
        flags=re.DOTALL,
    )

    # cases: "f of x equals k if x greater than 0, and 0 otherwise"
    def _cases(m: re.Match) -> str:
        body = m.group(1)
        rows = [r.strip() for r in body.split(r"\\") if r.strip()]
        arms = []
        for i, row in enumerate(rows):
            mm = re.match(r"([^&]*)&\s*(.*)", row)
            if mm:
                value, cond = mm.group(1).strip(), mm.group(2).strip()
                arms.append(f"{value} {cond}")
            else:
                arms.append(row)
        conj = ["that is", "and"][: max(1, len(arms))]
        parts = [f"{conj[0]} {arms[0]}"] + [f"and {a}" for a in arms[1:]]
        return " the function " + ", ".join(parts) + " "

    s = re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", _cases, s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{(\w+)\}", " ", s)
    s = re.sub(r"\\end\{(\w+)\}", " ", s)

    # accents: \hat{x} -> "x hat", etc. (mainstream spoken convention)
    s = re.sub(r"\\hat\{([^{}]*)\}", r" \1 hat ", s)
    s = re.sub(r"\\tilde\{([^{}]*)\}", r" \1 tilde ", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", r" \1 bar ", s)
    s = re.sub(r"\\vec\{([^{}]*)\}", r" vector \1 ", s)
    s = re.sub(r"\\dot\{([^{}]*)\}", r" \1 dot ", s)
    s = re.sub(r"\\ddot\{([^{}]*)\}", r" \1 double dot ", s)
    # \mathbf{v} -> v (boldness is not spoken)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r" \1 ", s)
    # \mathbb{R} -> the real numbers, etc.
    _BB = {"R": "real numbers", "N": "natural numbers", "Z": "integers",
           "Q": "rational numbers", "C": "complex numbers", "E": "expected value"}
    s = re.sub(r"\\mathbb\{(\w)\}", lambda m: f" the {_BB.get(m.group(1), m.group(1) + ' set')} ", s)
    s = re.sub(r"\\mathcal\{(\w)\}", lambda m: f" calligraphic {m.group(1)} ", s)
    # \binom{n}{k} -> n choose k
    s = re.sub(r"\\(?:t)?binom" + _NESTED + _NESTED, r" \1 choose \2 ", s)
    # \pmod{n} / \bmod n  (run before \pm operator can eat the prefix)
    s = re.sub(r"\\(?:p)?b?mod\s*\{?([^{}]*)\}?", r" modulo \1 ", s)
    # divergence / gradient ops
    s = re.sub(r"\\nabla\s*\\cdot", " the divergence of ", s)
    s = re.sub(r"\\nabla\s*\\times", " the curl of ", s)

    # greek letters (longest names first to avoid partial matches)
    for name in sorted(GREEK, key=len, reverse=True):
        s = s.replace("\\" + name, " " + GREEK[name] + " ")

    # functions
    for cmd in sorted(FUNCTIONS, key=len, reverse=True):
        s = s.replace(cmd, FUNCTIONS[cmd] + " ")

    # ellipsis variants BEFORE operators (\cdots must not be eaten by \cdot)
    for cmd, spoken in sorted(_DOTS.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(cmd, spoken)

    # operators (longest first)
    for cmd in sorted(OPERATORS, key=len, reverse=True):
        s = s.replace(cmd, OPERATORS[cmd])

    # degree / primes for angles: skip; macros we do not know -> drop the backslash, keep word
    s = re.sub(r"\\(\w+)", r" \1 ", s)
    s = s.replace("\\", " ")

    # character-level pass for remaining ASCII math
    out = []
    for ch in s:
        out.append(_spell_symbol(ch) if ch in "+-*=<>()[]{}/|!^.:" else ch)
    s = "".join(out)

    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.])", r"\1", s)
    # dangling pauses: commas right after "of"/"the ... of" or doubled commas
    s = re.sub(r"\bof\s*,", " of", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Convenience: symbol -> spoken word for the formula generator's symbol pool
SYMBOL_POOL = ["x", "y", "z", "a", "b", "c", "n", "t", "u", "v", "f", "g"]
GREEK_POOL = [r"\alpha", r"\beta", r"\theta", r"\lambda", r"\mu", r"\sigma", r"\phi", r"\omega"]
