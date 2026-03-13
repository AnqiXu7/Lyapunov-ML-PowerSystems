# ============================================================
# 06_extract.py
# Rational Lyapunov 4D double-machine vs infinite bus generator
# EXPANSION MODE
#
# Generate symbolic datasets for learning rational Lyapunov
# functions from nonlinear power system dynamics.
#
# Output format:
#
#   1| INT+ 4 <SPECIAL_3> f0 <SPECIAL_3> f1 <SPECIAL_3> f2 <SPECIAL_3> f3 \t V
#
# where
#   f_i : system dynamics
#   V   : rational Lyapunov function
#
# ============================================================

import random
import sympy as sp
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

SEP = "<SPECIAL_3>"
STATE_DIM = 4

# generate 10000 samples
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
        sign_str = "INT+" if iv >= 0 else "INT-"

        return f"{sign_str} {encode_base1000(iv)}"

    if fv == 0:
        return "FLOAT+ 0 10^ INT+ 0"

    sign_str = "FLOAT+" if fv > 0 else "FLOAT-"
    fv = abs(fv)

    s = "{:.4e}".format(fv)
    mantissa, exponent = s.split("e")

    mantissa_int = int(mantissa.replace(".", ""))

    mantissa_tokens = encode_base1000(mantissa_int)

    exp_val = int(exponent)
    exp_sign = "INT+" if exp_val >= 0 else "INT-"

    return f"{sign_str} {mantissa_tokens} 10^ {exp_sign} {abs(exp_val)}"


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
            return ["^"] + rec(e.args[0]) + rec(e.args[1])

        if e.func == sp.sin:
            return ["sin"] + rec(e.args[0])

        if e.func == sp.cos:
            return ["cos"] + rec(e.args[0])

        return [str(e)]

    return " ".join(rec(expr))

# ============================================================
# Symbolic variables
# ============================================================

x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3", real=True)

# ============================================================
# Parameter sampling
# ============================================================

def sample_parameters():

    theta1_star = random.uniform(0.42, 0.52)
    theta3_star = random.uniform(0.42, 0.52)

    c10 = random.uniform(30, 37)
    c11 = random.uniform(1.6, 2.1)
    c12 = random.uniform(4.8, 5.8)
    c13 = random.uniform(54, 64)
    c14 = random.uniform(15, 19)
    c15 = random.uniform(1.6, 2.1)

    c20 = random.uniform(43, 53)
    c21 = random.uniform(10, 13)
    c22 = random.uniform(2.9, 3.6)
    c23 = random.uniform(90, 110)
    c24 = random.uniform(1.0, 1.5)
    c25 = random.uniform(1.0, 1.5)

    return locals()

# ============================================================
# Build one dataset sample
# ============================================================

def build_sample_line():

    p = sample_parameters()

    theta1 = x0 + p["theta1_star"]
    theta3 = x2 + p["theta3_star"]

    theta13 = theta1 - theta3

    f0 = x1

    f1 = (
        p["c10"]
        - p["c11"] * sp.cos(theta13)
        - p["c12"] * sp.cos(theta1)
        - p["c13"] * sp.sin(theta1)
        - p["c14"] * sp.sin(theta13)
        - p["c15"] * x1
    )

    f2 = x3

    f3 = (
        p["c20"]
        + p["c21"] * sp.sin(theta13)
        - p["c22"] * sp.cos(theta3)
        - p["c23"] * sp.sin(theta3)
        - p["c24"] * sp.cos(theta13)
        - p["c25"] * x3
    )

    f_list = [sp.simplify(f0), sp.simplify(f1), sp.simplify(f2), sp.simplify(f3)]

    # rational Lyapunov
    num = x0**2 + x1**2 + x2**2 + x3**2 + x0**4 + x0**2*x2**2 + x2**4
    den = 1 + 2*x0 + x1 - 2*x2 + 8*x0**2 + 4*x1**2 + 4*x2**2

    V_expr = sp.simplify(num / den)

    for expr in f_list + [V_expr]:

        s = str(expr)

        if "zoo" in s or "nan" in s or "oo" in s:
            return None

    original_tokens = [expr_to_prefix_no_end(fi) for fi in f_list]

    dim_prefix = f"INT+ {encode_base1000(STATE_DIM)} {SEP} "

    original_joined = dim_prefix + f" {SEP} ".join(original_tokens)

    V_tokens = expr_to_prefix_no_end(V_expr)

    line = "1|" + original_joined + "\t" + V_tokens + "\n"

    return line

# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_total):

    lines = []
    attempts = 0

    pbar = tqdm(total=n_total, desc="Generating 06 samples")

    while len(lines) < n_total:

        attempts += 1

        line = build_sample_line()

        if line is not None:

            lines.append(line)
            pbar.update(1)

    pbar.close()

    print(f"Total attempts: {attempts}")

    return lines

# ============================================================
# Save dataset
# ============================================================

def save_all(lines):

    with open("PS_06_all", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\nSaved dataset:")
    print(f"PS_06_all : {len(lines)} samples")

# ============================================================
# Main
# ============================================================

def main():

    print("Generating rational Lyapunov dataset (06)")
    print(f"Target samples: {N_TOTAL}")

    lines = generate_dataset(N_TOTAL)

    save_all(lines)

    print("\nDone.")

if __name__ == "__main__":
    main()
