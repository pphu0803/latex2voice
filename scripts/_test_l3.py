import sys
sys.path.insert(0, ".")
from src.latextts.rules_en import latex_to_spoken as L

BS = chr(92)  # backslash
DB = BS + BS  # double backslash (latex row separator)

tests = [
    r"\begin{pmatrix} 1 & 2 %s 3 & 4 \end{pmatrix}" % DB,
    "f(x) = \\begin{cases} 5 & \\text{if } x > 0 %s 0 & \\text{otherwise} \\end{cases}" % DB,
    r"\left\| x \right\|^2 = x^\top x",
    r"\frac{\partial L}{\partial w} = \sum_i \delta_i x_i",
    r"A = \begin{bmatrix} a & b %s c & d \end{bmatrix}" % DB,
]
for t in tests:
    print(t)
    print("  ->", L(t), "\n")
