# ============================================================
# 01_extract.py
# Kundur 3-generator reduced power system dataset generator
# ------------------------------------------------------------
#
# This script generates symbolic datasets for learning Lyapunov
# functions from nonlinear power system dynamics.
#
# Key features:
#   - Uses a 3-generator Kundur-style reduced power system model
#   - Computes Lyapunov candidates via LMI optimization
#   - Converts symbolic expressions to prefix token sequences
#   - Filters invalid expressions (nan / inf / unexpected symbols)
#   - Generates a single expanded dataset file: PS_Kundur_all
#
# Output format:
#
#   1| INT+ 6 <SPECIAL_3> f0 <SPECIAL_3> f1 ... <SPECIAL_3> f5 \t V
#
# where:
#   f_i : system dynamics expressions
#   V   : Lyapunov function expression
#
# ============================================================

import random
import numpy as np
import cvxpy as cp
import sympy as sp
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

SEP = "<SPECIAL_3>"
STATE_DIM = 6

# Total number of valid samples to generate
N_TOTAL = 10000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Tokenization utilities
# These functions convert numeric values and symbolic
# expressions into prefix tokens compatible with the
# symbolic transformer training pipeline.
# ============================================================
def build_sample_line():
    """
    Build one valid Kundur sample using the same compact-form
    structure as the original 01.py, but with randomly sampled
    parameters from sample_case().
    """
    try:
        # ------------------------------------------------------------
        # 1) Sample one perturbed case
        # ------------------------------------------------------------
        case = sample_case()

        V1 = case["V1"]
        V2 = case["V2"]
        V3 = case["V3"]
        B12 = case["B12"]
        B13 = case["B13"]
        B23 = case["B23"]
        delta_star_np = case["delta_star"]
        M_np = case["M_np"]
        D_np = case["D_np"]

        Minv_np = np.linalg.inv(M_np)

        # ------------------------------------------------------------
        # 2) Build compact-form matrices exactly following 01.py
        # ------------------------------------------------------------
        b12 = B12 * V1 * V2
        b13 = B13 * V1 * V3
        b23 = B23 * V2 * V3
        Gamma_np = np.diag([b12, b13, b23])

        E_np = np.array([
            [1, -1,  0],
            [1,  0, -1],
            [0,  1, -1],
        ], dtype=float)

        Z3 = np.zeros((3, 3))
        I3 = np.eye(3)

        A_np = np.block([
            [Z3, I3],
            [Z3, -Minv_np @ D_np],
        ])

        C_np = np.hstack([E_np, np.zeros((3, 3))])

        B_np = np.vstack([
            np.zeros((3, 3)),
            Minv_np @ E_np.T @ Gamma_np
        ])

        # ------------------------------------------------------------
        # 3) Solve LMI
        # ------------------------------------------------------------
        sol = solve_lmi(A_np, B_np, C_np)
        if sol is None:
            return None

        Q_num = sol["Q_num"]
        k_num = sol["k_num"]

        # ------------------------------------------------------------
        # 4) Build symbolic dynamics exactly following 01.py
        # ------------------------------------------------------------
        x = sp.symbols("x0:6")
        x_delta = sp.Matrix(x[:3])
        omega = sp.Matrix(x[3:])

        E = sp.Matrix(E_np)
        Gamma = sp.Matrix(Gamma_np)
        delta_star = sp.Matrix(delta_star_np)

        M = sp.Matrix(M_np)
        D = sp.Matrix(D_np)
        Minv = M.inv()

        edge_angle = E * (x_delta + delta_star)
        edge_angle_star = E * delta_star

        F = sp.Matrix([
            sp.sin(edge_angle[i]) - sp.sin(edge_angle_star[i])
            for i in range(3)
        ])

        xdot_delta = omega
        xdot_omega = -(Minv * D) * omega - (Minv * E.T * Gamma) * F

        f_vec = sp.Matrix.vstack(xdot_delta, xdot_omega)
        f_list = [f_vec[i] for i in range(6)]

        # ------------------------------------------------------------
        # 5) Build Lyapunov function exactly following 01.py
        # ------------------------------------------------------------
        Q = sp.Matrix(Q_num)
        k = sp.Matrix(k_num)

        y = E * x_delta
        delta_e = edge_angle_star + y

        V_quad = sp.Rational(1, 2) * (sp.Matrix(x).T * Q * sp.Matrix(x))[0]

        V_nl = 0
        for i in range(3):
            V_nl += k[i] * (
                sp.cos(delta_e[i]) +
                delta_e[i] * sp.sin(edge_angle_star[i])
            )

        V_expr = sp.expand(V_quad - V_nl)

        # ------------------------------------------------------------
        # 6) Validate expressions
        # ------------------------------------------------------------
        ok, unexpected = has_only_allowed_symbols(f_list + [V_expr])
        if not ok:
            return None

        if not is_good_numeric_expr(f_list + [V_expr]):
            return None

        # ------------------------------------------------------------
        # 7) Tokenize
        # Format:
        # 1| INT+ 6 <SPECIAL_3> f0 <SPECIAL_3> ... <SPECIAL_3> f5 \t V
        # ------------------------------------------------------------
        orig_tokens = [expr_to_prefix_no_end(fi) for fi in f_list]
        dim_prefix = f"{num_token(STATE_DIM)} {SEP} "
        orig_joined = dim_prefix + f" {SEP} ".join(orig_tokens)

        v_tokens = expr_to_prefix_no_end(V_expr)

        line = "1| " + orig_joined + "\t" + v_tokens + "\n"
        return line

    except Exception as e:
        print("Sample generation error:", e)
        return None

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

    Integer example:
        5  -> INT+ 5
        -7 -> INT- 7

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
# Parameter sampling
# Randomly sample operating points around a nominal
# Kundur-style operating condition.
# ============================================================

def sample_case():

    V1 = random.uniform(1.03, 1.08)
    V2 = random.uniform(1.02, 1.07)
    V3 = random.uniform(1.00, 1.04)

    B12 = random.uniform(0.9 * 0.739, 1.1 * 0.739)
    B13 = random.uniform(0.9 * 1.0958, 1.1 * 1.0958)
    B23 = random.uniform(0.9 * 1.245, 1.1 * 1.245)

    delta1 = random.uniform(-0.15, -0.05)
    delta2 = random.uniform(0.02, 0.10)
    delta3 = 0.0

    m1 = random.uniform(1.5, 3.0)
    m2 = random.uniform(1.5, 3.0)
    m3 = random.uniform(1.5, 3.0)

    d1 = random.uniform(0.5, 2.0)
    d2 = random.uniform(0.5, 2.0)
    d3 = random.uniform(0.5, 2.0)

    return {
        "V1": V1, "V2": V2, "V3": V3,
        "B12": B12, "B13": B13, "B23": B23,
        "delta_star": np.array([delta1, delta2, delta3], dtype=float),
        "M_np": np.diag([m1, m2, m3]),
        "D_np": np.diag([d1, d2, d3]),
    }

# ============================================================
# LMI solver for Lyapunov function synthesis
# ============================================================

def solve_lmi(A_np, B_np, C_np):

    n = 6
    m = 3
    eps = 1e-6

    Q = cp.Variable((n, n), symmetric=True)
    k = cp.Variable(m, nonneg=True)
    h = cp.Variable(m, nonneg=True)

    K = cp.diag(k)
    H = cp.diag(h)

    R = Q @ B_np - C_np.T @ H - (K @ C_np @ A_np).T

    LMI = cp.bmat([
        [A_np.T @ Q + Q @ A_np, R],
        [R.T, -2 * H]
    ])

    constraints = [
        Q >> eps * np.eye(n),
        k >= eps,
        h >= eps,
        LMI << 0,
        cp.trace(Q) == 1.0
    ]

    prob = cp.Problem(cp.Maximize(cp.sum(k)), constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        return None

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    if Q.value is None:
        return None

    return {
        "Q_num": Q.value,
        "k_num": k.value,
        "h_num": h.value,
    }

# ============================================================
# Expression validation
# ============================================================

def has_only_allowed_symbols(exprs):

    allowed = {f"x{i}" for i in range(6)}

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
# Dataset generation
# ============================================================

def generate_dataset(n_total):

    lines = []
    attempts = 0
    max_attempts = n_total * 50

    pbar = tqdm(total=n_total, desc="Generating Kundur samples")

    while len(lines) < n_total and attempts < max_attempts:

        attempts += 1

        line = build_sample_line()

        if line is not None:
            lines.append(line)
            pbar.update(1)

    pbar.close()

    return lines

# ============================================================
# Save dataset
# ============================================================

def save_all(lines):

    with open("PS_Kundur_all", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("\nDataset saved:")
    print(f"PS_Kundur_all : {len(lines)} samples")

# ============================================================
# Main entry
# ============================================================

def main():

    print("Generating Kundur power system dataset")
    print(f"Target samples: {N_TOTAL}")

    lines = generate_dataset(N_TOTAL)

    save_all(lines)

    if lines:
        print("\nExample sample:\n")
        print(lines[0][:800])


if __name__ == "__main__":
    main()
