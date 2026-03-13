# ============================================================
# 07_extract.py
# VSG Lyapunov dataset generator
# EXPANSION MODE
#
# Generate symbolic datasets for learning Lyapunov functions
# from Virtual Synchronous Generator (VSG) nonlinear dynamics.
#
# Output format:
#
#   1| INT+ 2 <SPECIAL_3> f0 <SPECIAL_3> f1 \t V
#
# where
#   f0,f1 : system dynamics
#   V     : Lyapunov function
#
# ============================================================

import random
import math
import sympy as sp
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

SEP = "<SPECIAL_3>"
STATE_DIM = 2

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

    return " ".join(rec(sp.simplify(expr)))

# ============================================================
# Symbolic variables
# ============================================================

x0, x1 = sp.symbols("x0 x1", real=True)

# ============================================================
# Symbolic system definition
# ============================================================

J, D = sp.symbols("J D", positive=True)
T0 = sp.symbols("T0")
E, Vg = sp.symbols("E Vg", positive=True)
B, G = sp.symbols("B G")
omega0 = sp.symbols("omega0", positive=True)
delta_s = sp.symbols("delta_s")
lam = sp.symbols("lam", positive=True)

# dynamics
f0_sym = x1

f1_sym = (
    T0
    - (E * Vg * B / omega0) * sp.sin(delta_s + x0)
    - (E * Vg * G / omega0) * sp.cos(delta_s + x0)
    - D * x1
) / J

f_list_sym = [f0_sym, f1_sym]

# Lyapunov
V_sym = (
    sp.Rational(1, 2) * J * x1**2
    - T0 * x0
    + (E * Vg * B / omega0) * (sp.cos(delta_s + x0) - sp.cos(delta_s))
    - (E * Vg * G / omega0) * (sp.sin(delta_s + x0) - sp.sin(delta_s))
    + D * lam * x0 * x1
    + (D**2 / (2 * J)) * lam * x0**2
)

# ============================================================
# Parameter sampling
# ============================================================

def sample_parameters():

    omega0_num = 314.159

    E_num = random.uniform(280, 340)
    Vg_num = random.uniform(280, 340)

    R_num = random.uniform(0.05, 0.2)
    X_num = random.uniform(0.15, 0.4)

    denom = R_num**2 + X_num**2

    G_num = R_num / denom
    B_num = -X_num / denom

    J_num = random.uniform(3, 8)
    D_num = random.uniform(2, 12)
    lam_num = random.uniform(0.2, 1.0)

    delta_s_num = random.uniform(0.2, 0.8)

    P0_num = random.uniform(10000, 30000)
    T0_num = P0_num / omega0_num

    return {
        J: J_num,
        D: D_num,
        T0: T0_num,
        E: E_num,
        Vg: Vg_num,
        omega0: omega0_num,
        G: G_num,
        B: B_num,
        lam: lam_num,
        delta_s: delta_s_num,
    }

# ============================================================
# Build one dataset sample
# ============================================================

def build_sample_line(subs_numeric):

    f_list = [sp.simplify(fi.subs(subs_numeric)) for fi in f_list_sym]
    V_expr = sp.simplify(V_sym.subs(subs_numeric))

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

    pbar = tqdm(total=n_total, desc="Generating VSG samples")

    while len(lines) < n_total:

        attempts += 1

        subs_numeric = sample_parameters()

        line = build_sample_line(subs_numeric)

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

    with open("PS_VSG_all", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\nSaved dataset:")
    print(f"PS_VSG_all : {len(lines)} samples")

# ============================================================
# Main
# ============================================================

def main():

    print("Generating VSG Lyapunov dataset")
    print(f"Target samples: {N_TOTAL}")

    lines = generate_dataset(N_TOTAL)

    save_all(lines)

    print("\nDone.")

if __name__ == "__main__":
    main()
