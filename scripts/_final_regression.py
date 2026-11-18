"""Final v4 regression: stress set + guard."""
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL, USER_TMPL_PLAIN  # noqa: E402
from src.latextts.tn_infer import get_model  # noqa: E402

model, tok = get_model("cpu")


def raw(user_content: str) -> str:
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


tests = [
    (r"(a+b)^2 = a^2 + 2ab + b^2", "GROUPING"),
    (r"\left( x + 1 \right)^3", "GROUPING"),
    (r"2(u-a)^2", "GROUPING"),
    (r"\sum_{i \in S} w_i x_i", "OOD"),
    (r"\max_{\theta} \mathbb{E}_{x \sim p}[\log p_{\theta}(x)]", "OOD"),
    (r"\frac{\partial^2 f}{\partial x \partial y}", "OOD"),
    (r"x \in \mathbb{R}^{n \times d}", "OOD"),
    (r"\int_a^b f(g(t)) g^\prime(t)\, dt", "OOD"),
    (r"\begin{pmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \\ 0 & 4 & 5 \end{pmatrix}", "MATRIX"),
]
for latex, tag in tests:
    print(f"[{tag}] {latex}", flush=True)
    print(f"  -> {raw(USER_TMPL.format(latex=latex))}", flush=True)

s = "The experiment shows a clear improvement."
print(f"[GUARD] {s!r}", flush=True)
print(f"  -> {raw(USER_TMPL_PLAIN.format(text=s))!r}", flush=True)
