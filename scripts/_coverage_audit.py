"""Coverage audit: common math/physics symbols vs rules library.

Each symbol is fed through latex_to_spoken and the output classified by
expected-keyword match. Rules output = training gold, so this table =
training-set coverage.
"""
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from src.latextts.rules_en import latex_to_spoken as L

CASES = [
    ("greek-lc", r"\alpha", "alpha"), ("greek-lc", r"\beta", "beta"),
    ("greek-lc", r"\gamma", "gamma"), ("greek-lc", r"\delta", "delta"),
    ("greek-lc", r"\epsilon", "epsilon"), ("greek-lc", r"\varepsilon", "epsilon"),
    ("greek-lc", r"\zeta", "zeta"), ("greek-lc", r"\eta", "eta"),
    ("greek-lc", r"\theta", "theta"), ("greek-lc", r"\iota", "iota"),
    ("greek-lc", r"\kappa", "kappa"), ("greek-lc", r"\lambda", "lambda"),
    ("greek-lc", r"\mu", "mu"), ("greek-lc", r"\nu", "nu"),
    ("greek-lc", r"\xi", "xi"), ("greek-lc", r"\pi", "pi"),
    ("greek-lc", r"\rho", "rho"), ("greek-lc", r"\sigma", "sigma"),
    ("greek-lc", r"\tau", "tau"), ("greek-lc", r"\upsilon", "upsilon"),
    ("greek-lc", r"\phi", "phi"), ("greek-lc", r"\varphi", "phi"),
    ("greek-lc", r"\chi", "kai"), ("greek-lc", r"\psi", "psi"),
    ("greek-lc", r"\omega", "omega"),
    ("greek-cap", r"\Gamma", "capital gamma"), ("greek-cap", r"\Delta", "capital delta"),
    ("greek-cap", r"\Theta", "capital theta"), ("greek-cap", r"\Lambda", "capital lambda"),
    ("greek-cap", r"\Xi", "capital xi"), ("greek-cap", r"\Pi", "capital pi"),
    ("greek-cap", r"\Sigma", "capital sigma"), ("greek-cap", r"\Phi", "capital phi"),
    ("greek-cap", r"\Psi", "capital psi"), ("greek-cap", r"\Omega", "capital omega"),
    ("relation", r"x \approx y", "approximately"), ("relation", r"x \neq y", "not equal"),
    ("relation", r"x \leq y", "less than or equal"), ("relation", r"x \geq y", "greater than or equal"),
    ("relation", r"x \propto y", "proportional"), ("relation", r"x \equiv y", "identically equal"),
    ("relation", r"a \in S", "is in"), ("relation", r"a \notin S", "not in"),
    ("relation", r"A \subset B", "subset"), ("relation", r"A \subseteq B", "subset"),
    ("relation", r"A \cup B", "union"), ("relation", r"A \cap B", "intersect"),
    ("relation", r"f \circ g", "composed with"),
    ("logic", r"\forall x", "for all"), ("logic", r"\exists x", "there exists"),
    ("logic", r"P \Rightarrow Q", "implies"), ("logic", r"P \leftrightarrow Q", "if and only if"),
    ("logic", r"\emptyset", "empty set"), ("logic", r"\infty", "infinity"),
    ("arrow", r"f: A \to B", "to"), ("arrow", r"x \mapsto x^2", "maps to"),
    ("struct", r"\frac{a}{b}", "over"), ("struct", r"\sqrt{x}", "square root"),
    ("struct", r"\sqrt[3]{x}", "root"), ("struct", r"x^{n+1}", "power"),
    ("struct", r"x_i", "sub"), ("struct", r"\binom{n}{k}", "choose"),
    ("struct", r"n \bmod 3", "modulo"),
    ("calc", r"\int_0^1 f(x) dx", "integral"), ("calc", r"\oint_C", "oint"),
    ("calc", r"\sum_{i=1}^n", "sum"), ("calc", r"\prod_{i=1}^n", "product"),
    ("calc", r"\lim_{n \to \infty}", "limit"), ("calc", r"\frac{df}{dx}", "df over dx"),
    ("calc", r"\frac{\partial f}{\partial x}", "partial"), ("calc", r"\nabla f", "nabla"),
    ("calc", r"\nabla \cdot F", "divergence"),
    ("fn", r"\sin x", "sine"), ("fn", r"\cos x", "cosine"), ("fn", r"\tan x", "tangent"),
    ("fn", r"\cot x", "cotangent"), ("fn", r"\sec x", "secant"), ("fn", r"\csc x", "cosecant"),
    ("fn", r"\arcsin x", "arc sine"), ("fn", r"\sinh x", "hyperbolic sine"),
    ("fn", r"\log_2 n", "log"), ("fn", r"\ln x", "natural log"), ("fn", r"\exp x", "exp"),
    ("fn", r"\max_x", "max"), ("fn", r"\min_x", "min"), ("fn", r"\det A", "determinant"),
    ("fn", r"\sup S", "supremum"), ("fn", r"\inf S", "infimum"),
    ("fn", r"\gcd(a,b)", "greatest common divisor"),
    ("physics", r"\hbar", "h bar"), ("physics", r"\vec{v}", "vector v"),
    ("physics", r"\hat{x}", "x hat"), ("physics", r"\tilde{f}", "f tilde"),
    ("physics", r"\bar{x}", "x bar"), ("physics", r"\dot{x}", "x dot"),
    ("physics", r"\ddot{x}", "x double dot"), ("physics", r"\|v\|", "norm"),
    ("physics", r"45^\circ", "degrees"),
    ("physics", r"\mathrm{kg}", "kg"), ("physics", r"\mathrm{m/s}", "over"),
    ("font", r"\mathbb{R}", "real numbers"), ("font", r"\mathcal{L}", "calligraphic"),
    ("font", r"\mathbf{v}", "v"), ("font", r"\text{if } x > 0", "if"),
    ("abbr", r"a \equiv b \pmod{n}", "modulo"), ("abbr", r"\text{const}", "const"),
    ("abbr", r"i \dots j", "dots"),
    # --- amsmath official \DeclareMathOperator list (30 names, amsmath v2 sec 5) ---
    ("amsmath", r"\arccos x", "arc cosine"),
    ("amsmath", r"\arcsin x", "arc sine"), ("amsmath", r"\arctan x", "arc tangent"),
    ("amsmath", r"\arg z", "argument"), ("amsmath", r"\cos x", "cosine"),
    ("amsmath", r"\cosh x", "hyperbolic cosine"), ("amsmath", r"\cot x", "cotangent"),
    ("amsmath", r"\coth x", "hyperbolic cotangent"), ("amsmath", r"\csc x", "cosecant"),
    ("amsmath", r"\deg f", "degree"), ("amsmath", r"\det A", "determinant"),
    ("amsmath", r"\dim V", "dimension"), ("amsmath", r"\exp x", "exp"),
    ("amsmath", r"\hom(X,Y)", "hom"), ("amsmath", r"\ker A", "kernel"),
    ("amsmath", r"\lg n", "log base"), ("amsmath", r"\lim a_n", "limit"),
    ("amsmath", r"\liminf a_n", "limit inferior"), ("amsmath", r"\limsup a_n", "limit superior"),
    ("amsmath", r"\ln x", "natural log"), ("amsmath", r"\log x", "log"),
    ("amsmath", r"\max S", "max"), ("amsmath", r"\min S", "min"),
    ("amsmath", r"\Pr(X)", "probability"), ("amsmath", r"\sec x", "secant"),
    ("amsmath", r"\sin x", "sine"), ("amsmath", r"\sinh x", "hyperbolic sine"),
    ("amsmath", r"\sup S", "supremum"), ("amsmath", r"\tan x", "tangent"),
    ("amsmath", r"\tanh x", "hyperbolic tangent"),
    # --- named operators (no backslash macro; \operatorname / \mathrm passthrough) ---
    ("named", r"\operatorname{softmax}(z)", "softmax"),
    ("named", r"\operatorname{sigmoid}(z)", "sigmoid"),
    ("named", r"\operatorname{ReLU}(x)", "R E L U"),
    ("named", r"\mathrm{LayerNorm}(x)", "layer norm"),
    ("named", r"\mathrm{Attention}(Q,K,V)", "Attention"),
    ("named", r"\mathrm{Concat}(h_1,h_2)", "Concat"),
]

stats = defaultdict(lambda: [0, 0])
fails = []
for cat, latex, expect in CASES:
    out = L(latex)
    ok = all(re.search(p, out, re.I) for p in expect.split("|"))
    stats[cat][1] += 1
    if ok:
        stats[cat][0] += 1
    else:
        fails.append((cat, latex, expect, out))

print(f"{'category':12} {'covered':>8}")
for cat, (okc, tot) in stats.items():
    print(f"{cat:12} {okc:>3}/{tot:<4}")
print(f"\nTOTAL {sum(v[0] for v in stats.values())}/{sum(v[1] for v in stats.values())}\n")
print("=== GAPS ===")
for cat, latex, expect, out in fails:
    print(f"[{cat}] {latex!r}\n   want: {expect}\n   got : {out!r}")
