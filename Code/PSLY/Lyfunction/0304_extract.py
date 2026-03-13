# ============================================================
# 0304_extract.py
# 4D dataset generator (EXPANSION MODE)
# ------------------------------------------------------------
# Output format:
#   1| INT+ 4 <SPECIAL_3> f0 <SPECIAL_3> f1 <SPECIAL_3> f2 <SPECIAL_3> f3 \t V
#
# This version:
#   - DOES NOT split dataset
#   - outputs one file: PS_0304_all
#   - includes tqdm progress bar
#   - includes FLOAT/INT tokenization
# ============================================================

import random
import sympy as sp
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

SEP = "<SPECIAL_3>"
STATE_DIM = 4

N_TOTAL = 10000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# Tokenization utilities
# ============================================================

def encode_base1000(n):
    if n == 0:
        return "0"

    chunks = []
    n = abs(int(n))

    while n > 0:
        chunks.append(str(n % 1000))
        n //= 1000

    return " ".join(reversed(chunks))


def num_token(v):
    fv = float(v)

    if abs(fv - round(fv)) < 1e-12:
        iv = int(round(fv))
        sign = "INT+" if iv >= 0 else "INT-"
        return f"{sign} {encode_base1000(iv)}"

    if fv == 0:
        return "FLOAT+ 0 10^ INT+ 0"

    sign = "FLOAT+" if fv > 0 else "FLOAT-"
    fv = abs(fv)

    s = "{:.4e}".format(fv)
    mantissa, exponent = s.split("e")

    mantissa_int = int(mantissa.replace(".", ""))
    mantissa_tokens = encode_base1000(mantissa_int)

    exp_val = int(exponent)
    exp_sign = "INT+" if exp_val >= 0 else "INT-"

    return f"{sign} {mantissa_tokens} 10^ {exp_sign} {abs(exp_val)}"


def expr_to_prefix_no_end(expr):

    def rec(e):

        e = sp.simplify(e)

        if e.is_Number:
            return num_token(e).split(" ")

        if e.is_Symbol:
            return [str(e)]

        if isinstance(e, sp.Add):
            args = list(e.args)
            toks = rec(args[0])
            for a in args[1:]:
                toks = ["+"] + toks + rec(a)
            return toks

        if isinstance(e, sp.Mul):
            args = list(e.args)
            toks = rec(args[0])
            for a in args[1:]:
                toks = ["*"] + toks + rec(a)
            return toks

        if isinstance(e, sp.Pow):
            base, exp = e.args
            return ["^"] + rec(base) + rec(exp)

        if e.func == sp.sin:
            return ["sin"] + rec(e.args[0])

        if e.func == sp.cos:
            return ["cos"] + rec(e.args[0])

        return [str(e)]

    return " ".join(rec(expr))


# ============================================================
# Symbolic variables
# ============================================================

x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")

# ============================================================
# Base system
# ============================================================

f0_base = -0.5*x0 + x1 - 0.2*x2**2
f1_base = -1.2*x1 + 0.5*x0*x2 - x3
f2_base = -2.0*x2 - 0.1*x0*x1 + 0.5*x3**2
f3_base = -1.5*x3 + x2

V_base = 2.5*x0**2 + 3.1*x1**2 + 1.8*x2**2 + 2.0*x3**2

# ============================================================
# Validation
# ============================================================

def has_only_allowed_symbols(exprs):

    allowed = {"x0", "x1", "x2", "x3"}

    expr_symbols = set()

    for expr in exprs:
        expr_symbols |= {str(s) for s in expr.free_symbols}

    unexpected = sorted(expr_symbols - allowed)

    return len(unexpected) == 0, unexpected


def is_good_numeric_expr(exprs):

    for expr in exprs:
        s = str(expr)

        if "zoo" in s or "nan" in s or "oo" in s:
            return False

    return True


# ============================================================
# Build sample
# ============================================================

def build_sample_line():

    alpha = random.uniform(0.5, 2.0)
    beta = random.uniform(0.5, 2.0)

    f0 = sp.expand(alpha * f0_base)
    f1 = sp.expand(alpha * f1_base)
    f2 = sp.expand(alpha * f2_base)
    f3 = sp.expand(alpha * f3_base)

    V = sp.expand(beta * V_base)

    exprs = [f0, f1, f2, f3, V]

    if not is_good_numeric_expr(exprs):
        return None

    ok, unexpected = has_only_allowed_symbols(exprs)

    if not ok:
        return None

    f0_tok = expr_to_prefix_no_end(f0)
    f1_tok = expr_to_prefix_no_end(f1)
    f2_tok = expr_to_prefix_no_end(f2)
    f3_tok = expr_to_prefix_no_end(f3)

    V_tok = expr_to_prefix_no_end(V)

    dim_prefix = f"INT+ {encode_base1000(STATE_DIM)} {SEP} "

    original = dim_prefix + f"{f0_tok} {SEP} {f1_tok} {SEP} {f2_tok} {SEP} {f3_tok}"

    line = f"1|{original}\t{V_tok}\n"

    return line


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_total):

    lines = []
    attempts = 0
    max_attempts = n_total * 20

    pbar = tqdm(total=n_total, desc="Generating 0304 samples")

    while len(lines) < n_total and attempts < max_attempts:

        attempts += 1

        line = build_sample_line()

        if line is not None:
            lines.append(line)
            pbar.update(1)

    pbar.close()

    if len(lines) < n_total:
        print(f"WARNING: generated only {len(lines)} samples")

    return lines


# ============================================================
# Save dataset
# ============================================================

def save_all(lines):

    with open("PS_0304_all", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nSaved PS_0304_all : {len(lines)} samples")


# ============================================================
# Main
# ============================================================

def main():

    print("Generating expanded 0304 dataset")
    print(f"Target size: {N_TOTAL}")

    lines = generate_dataset(N_TOTAL)

    save_all(lines)

    print("\nExample line:\n")
    print(lines[0][:500])


if __name__ == "__main__":
    main()
