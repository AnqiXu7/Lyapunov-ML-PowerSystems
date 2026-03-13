# ============================================================
# 02_extract.py
# Classical 2-bus system dataset generator
# EXPANSION MODE: generate one merged file only
# ------------------------------------------------------------
#
# This script generates symbolic datasets for learning
# Lyapunov functions from a classical 2-bus nonlinear system.
#
# Key features:
#   - Randomly samples physically consistent system parameters
#   - Builds system dynamics and Lyapunov expressions
#   - Converts expressions to prefix token format
#   - Filters invalid expressions
#   - Generates a single expanded dataset file: PS_2bus_all
#
# Output format:
#
#   1| INT+ 2 <SPECIAL_3> f0 <SPECIAL_3> f1 \t V
#
# where:
#   f0, f1 : system dynamics expressions
#   V      : Lyapunov function expression
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

# Total number of valid samples to generate
N_TOTAL = 10000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# Tokenization utilities
# These functions convert numbers and symbolic expressions
# into prefix tokens compatible with the training pipeline.
# ============================================================

def encode_base1000(n):
    """
    Convert an integer into base-1000 token representation.

    Example:
        1234567 -> "1 234 567"
    """
    if n == 0:
        return "0"

    chunks = []
    n = abs(int(n))

    while n > 0:
        chunks.append(str(n % 1000))
        n //= 1000

    return " ".join(reversed(chunks))


def num_token(v):
    """
    Convert numeric value into model token format.

    Integer examples:
        5   -> INT+ 5
        -12 -> INT- 12

    Float example:
        0.125 -> FLOAT+ 1250 10^ INT- 4
    """
    fv = float(v)

    # Integer case
    if abs(fv - round(fv)) < 1e-12:
        iv = int(round(fv))
        sign_str = "INT+" if iv >= 0 else "INT-"
        return f"{sign_str} {encode_base1000(iv)}"

    # Zero float
    if fv == 0:
        return "FLOAT+ 0 10^ INT+ 0"

    # General float
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
    """
    Convert a SymPy expression into prefix notation tokens.
    """

    def rec(e):
        if e.is_Number:
            return num_token(e).split()

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

    return " ".join(rec(sp.simplify(expr)))

# ============================================================
# Symbolic variables
# ============================================================

x0, x1 = sp.symbols("x0 x1", real=True)

# ============================================================
# Parameter sampling
# ============================================================

def sample_parameters():
    """
    Sample a self-consistent classical 2-bus system.

    The equilibrium condition is enforced by:
        p = k * sin(delta_star)

    so that x0 = 0, x1 = 0 is an equilibrium point.
    """
    d_num = random.uniform(0.5, 2.0)
    k_num = random.uniform(0.5, 1.5)
    delta_star_num = random.uniform(0.2, 0.8)
    p_num = k_num * math.sin(delta_star_num)

    return d_num, k_num, delta_star_num, p_num

# ============================================================
# Build one sample line
# ============================================================

def build_sample_line():
    """
    Build one valid dataset sample.
    """
    d_num, k_num, delta_star_num, p_num = sample_parameters()

    delta_star = sp.Float(delta_star_num)
    cos_delta_star = sp.Float(math.cos(delta_star_num))
    sin_delta_star = sp.Float(math.sin(delta_star_num))

    # System dynamics
    f0 = x1
    f1 = -sp.Float(d_num) * x1 - sp.Float(k_num) * sp.sin(x0 + delta_star) + sp.Float(p_num)

    f_list = [sp.simplify(f0), sp.simplify(f1)]

    # Lyapunov candidate
    V_expr = (
        sp.Float("0.5") * x1**2
        + sp.Float(k_num) * (
            cos_delta_star - sp.cos(x0 + delta_star) - x0 * sin_delta_star
        )
    )
    V_expr = sp.simplify(V_expr)

    # Check allowed symbols
    allowed_symbols = {"x0", "x1"}
    expr_symbols = set()
    for expr in f_list + [V_expr]:
        expr_symbols |= {str(s) for s in expr.free_symbols}

    unexpected = sorted(expr_symbols - allowed_symbols)
    if unexpected:
        print(f"[skip] unexpected symbols: {unexpected}")
        return None

    # Reject invalid numeric expressions
    for expr in f_list + [V_expr]:
        s = str(expr)
        if "zoo" in s or "nan" in s or "oo" in s:
            return None

    # Tokenize expressions
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
    """
    Generate n_total valid samples.

    Since invalid samples may be rejected, the number of attempts
    can exceed the number of valid generated lines.
    """
    lines = []
    attempts = 0
    max_attempts = n_total * 10

    pbar = tqdm(total=n_total, desc="Generating 2-bus samples")

    while len(lines) < n_total and attempts < max_attempts:
        attempts += 1
        line = build_sample_line()

        if line is not None:
            lines.append(line)
            pbar.update(1)

    pbar.close()

    if len(lines) < n_total:
        print(f"[warning] only generated {len(lines)} / {n_total} valid samples")

    print(f"Total attempts: {attempts}")
    return lines

# ============================================================
# Save dataset
# ============================================================

def save_all(lines):
    """
    Save all generated samples into one file.
    No train/valid/test split is used at this stage.
    """
    with open("PS_2bus_all", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\nDataset saved:")
    print(f"PS_2bus_all : {len(lines)} samples")

# ============================================================
# Main entry
# ============================================================

def main():
    print("Generating classical 2-bus Lyapunov dataset")
    print(f"Target samples: {N_TOTAL}")
    print("Mode: expansion only (no split)")

    lines = generate_dataset(N_TOTAL)
    save_all(lines)

    if lines:
        print("\nExample sample:\n")
        print(lines[0][:500])

    print("\nDone.")

if __name__ == "__main__":
    main()
