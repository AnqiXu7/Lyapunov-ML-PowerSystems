# ============================================================
# 05_extract.py
# Nonlinear 2D SMIB dataset generator using double algebraic scaling
# EXPANSION MODE: generate one merged file only
# ------------------------------------------------------------
#
# This script generates symbolic datasets for learning Lyapunov
# functions from a nonlinear single-machine infinite-bus (SMIB) system.
#
# Key features:
#   - Uses physically meaningful seeds from Table II
#   - Applies double algebraic scaling to increase diversity
#   - Converts symbolic expressions into prefix token sequences
#   - Generates a single expanded dataset file: PS_05_all
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

import sympy as sp
import random
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
        e = sp.simplify(e)

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
            return ["^"] + rec(e.args[0]) + rec(e.args[1])

        if e.func == sp.sin:
            return ["sin"] + rec(e.args[0])

        if e.func == sp.cos:
            return ["cos"] + rec(e.args[0])

        return [str(e)]

    return " ".join(rec(expr))

# ============================================================
# Physical constants and exact seed library
# These values come from the benchmark seed table.
# ============================================================

m = 0.1
d = 0.15
sigma = 0.2

TABLE_II = [
    {"delta_star": -0.253, "alpha": 0.273, "P11": 0.587, "P12": 0.354, "P22": 0.352},
    {"delta_star":  0.253, "alpha": 0.273, "P11": 0.587, "P12": 0.354, "P22": 0.352},
    {"delta_star":  0.524, "alpha": 0.275, "P11": 0.443, "P12": 0.273, "P22": 0.281},
    {"delta_star":  0.848, "alpha": 0.222, "P11": 0.219, "P12": 0.136, "P22": 0.160},
    {"delta_star":  1.120, "alpha": 0.138, "P11": 0.089, "P12": 0.063, "P22": 0.086},
    {"delta_star":  1.430, "alpha": 0.065, "P11": 0.027, "P12": 0.024, "P22": 0.038},
]

# ============================================================
# Expression validation
# ============================================================

def has_only_allowed_symbols(exprs):
    """
    Ensure that only x0 and x1 appear in expressions.
    """
    allowed = {"x0", "x1"}
    expr_symbols = set()

    for expr in exprs:
        expr_symbols |= {str(s) for s in expr.free_symbols}

    unexpected = sorted(expr_symbols - allowed)
    return len(unexpected) == 0, unexpected


def is_good_numeric_expr(exprs):
    """
    Reject expressions containing nan / inf / zoo.
    """
    for expr in exprs:
        s = str(expr)
        if "zoo" in s or "nan" in s or "oo" in s:
            return False
    return True

# ============================================================
# Build one sample line
# ============================================================

def build_sample_line(x0, x1):
    """
    Build one valid dataset sample using double algebraic scaling.
    """
    # Step 1: randomly select one physically valid seed
    case = random.choice(TABLE_II)

    delta_s = case["delta_star"]
    alpha_paper = case["alpha"]
    P11 = case["P11"]
    P12 = case["P12"]
    P22 = case["P22"]

    # Step 2: sample scaling coefficients
    # s1 scales the system dynamics
    # s2 scales the Lyapunov function
    s1 = random.uniform(0.1, 5.0)
    s2 = random.uniform(0.1, 5.0)

    # Step 3: construct and scale the nonlinear system
    f0_base = x1
    f1_base = -(d / m) * x1 - (sigma / m) * (sp.sin(x0 + delta_s) - sp.sin(delta_s))

    f0_new = sp.expand(s1 * f0_base)
    f1_new = sp.expand(s1 * f1_base)

    # Step 4: construct and scale the Lyapunov function
    kinetic = P11 * x0**2 + 2 * P12 * x0 * x1 + P22 * x1**2
    potential = 2 * alpha_paper * (
        sp.cos(delta_s) - sp.cos(x0 + delta_s) - x0 * sp.sin(delta_s)
    )

    V_base = kinetic + potential
    V_new = sp.expand(s2 * V_base)

    exprs = [f0_new, f1_new, V_new]

    # Step 5: validation
    if not is_good_numeric_expr(exprs):
        return None

    ok, unexpected = has_only_allowed_symbols(exprs)
    if not ok:
        print(f"[skip] unexpected symbols: {unexpected}")
        return None

    # Step 6: tokenize expressions
    f0_tok = expr_to_prefix_no_end(f0_new)
    f1_tok = expr_to_prefix_no_end(f1_new)
    V_tok = expr_to_prefix_no_end(V_new)

    dim_prefix = f"INT+ {encode_base1000(STATE_DIM)} {SEP} "
    original_function = dim_prefix + f"{f0_tok} {SEP} {f1_tok}"

    line = f"1|{original_function}\t{V_tok}\n"
    return line

# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_total):
    """
    Generate n_total valid samples.
    """
    x0, x1 = sp.symbols("x0 x1", real=True)

    lines = []
    attempts = 0
    max_attempts = n_total * 5

    pbar = tqdm(total=n_total, desc="Generating SMIB samples")

    while len(lines) < n_total and attempts < max_attempts:
        attempts += 1

        line = build_sample_line(x0, x1)

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
    with open("PS_05_all", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\nDataset saved:")
    print(f"PS_05_all : {len(lines)} samples")

# ============================================================
# Main entry
# ============================================================

def main():
    print("Generating nonlinear 2D SMIB Lyapunov dataset")
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
