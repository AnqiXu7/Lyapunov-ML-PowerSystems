# ============================================================
# 01_extract.py
# Kundur 3-generator reduced power system dataset generator
# ------------------------------------------------------------
#
# Physically constrained dataset expansion version
#
# Output format:
#   1| INT+ 6 <SPECIAL_3> f0 <SPECIAL_3> ... <SPECIAL_3> f5 \t V
#
# Main design goals:
#   1. Preserve the original paper compact-form structure
#   2. Perturb only physically meaningful parameters
#   3. Enforce angle-difference constraints motivated by the paper
#   4. Reject numerically or physically suspicious samples
#
# ============================================================

import math
import random
import warnings
import numpy as np
import cvxpy as cp
import sympy as sp
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

SEP = "<SPECIAL_3>"
STATE_DIM = 6

N_TOTAL = 10000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Nominal Kundur 3-generator reduced model parameters
# From paper / original 01.py
# ============================================================

V1_NOM, V2_NOM, V3_NOM = 1.0566, 1.0502, 1.0170
B12_NOM, B13_NOM, B23_NOM = 0.739, 1.0958, 1.245

# In original 01.py, delta3* = 0 is used as reference
DELTA_STAR_NOM = np.array([-0.1005, 0.0583, 0.0], dtype=float)

M_NOM = np.diag([2.0, 2.0, 2.0])
D_NOM = np.diag([1.0, 1.0, 1.0])

# Incidence matrix for edges: (1-2), (1-3), (2-3)
E_NP = np.array([
    [1.0, -1.0,  0.0],
    [1.0,  0.0, -1.0],
    [0.0,  1.0, -1.0],
], dtype=float)

# ============================================================
# Perturbation ranges
# Chosen to stay near the paper nominal case
# ============================================================

V_PERT_REL = 0.03       # ±3%
B_PERT_REL = 0.08       # ±8%
M_PERT_REL = 0.20       # ±20%
D_PERT_REL = 0.30       # ±30%
DELTA_PERT_ABS = 0.03   # ±0.03 rad

# ============================================================
# Physical constraints from the paper
# Paper polytope P: |delta_ij + delta_ij^*| < pi
# Stronger convex region Q: |delta_ij| < pi/2
#
# Here, for equilibrium sampling we enforce:
#   1) |delta_ij^*| < pi/2        (stronger, more conservative)
#   2) |delta_ij^*| < pi          (automatically implied)
#
# Since x = delta - delta*, keeping delta* in the convex region
# helps maintain physical realism around normal operating points.
# ============================================================

EQ_EDGE_LIMIT_MAIN = math.pi          # from polytope P
EQ_EDGE_LIMIT_CONVEX = math.pi / 2    # stronger convex-region constraint

MIN_EIG_Q = 1e-7
MIN_POSITIVE_VALUE = 1e-8

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

        if isinstance(e, sp.Function):
            return [e.func.__name__] + rec(e.args[0])

        return [str(e)]

    return " ".join(rec(sp.simplify(expr)))

# ============================================================
# Sampling helpers
# ============================================================

def sample_relative(nominal, rel_width):
    low = nominal * (1.0 - rel_width)
    high = nominal * (1.0 + rel_width)
    return random.uniform(low, high)


def sample_absolute(nominal, abs_width):
    return random.uniform(nominal - abs_width, nominal + abs_width)


def equilibrium_edges(delta_star):
    """
    Return equilibrium edge angle differences:
    [delta12*, delta13*, delta23*]
    """
    return E_NP @ delta_star


def is_physically_reasonable_case(case):
    V1, V2, V3 = case["V1"], case["V2"], case["V3"]
    B12, B13, B23 = case["B12"], case["B13"], case["B23"]
    delta_star = case["delta_star"]
    M_np = case["M_np"]
    D_np = case["D_np"]

    # positivity
    if min(V1, V2, V3) <= 0:
        return False
    if min(B12, B13, B23) <= 0:
        return False
    if np.min(np.diag(M_np)) <= 0:
        return False
    if np.min(np.diag(D_np)) <= 0:
        return False

    # generator 3 is the reference angle
    if abs(delta_star[2]) > 1e-12:
        return False

    # edge angle differences at equilibrium
    edge_star = equilibrium_edges(delta_star)

    # Main paper region: |delta_ij + delta_ij^*| < pi
    # At equilibrium x = 0, this reduces to |delta_ij^*| < pi.
    if np.max(np.abs(edge_star)) >= EQ_EDGE_LIMIT_MAIN:
        return False

    # Stronger convex-region constraint used in the paper discussion:
    # |delta_ij| < pi/2
    # Here we require equilibrium itself to lie inside this region.
    if np.max(np.abs(edge_star)) >= EQ_EDGE_LIMIT_CONVEX:
        return False

    return True


def sample_case():
    """
    Sample one physically constrained operating condition around
    the nominal Kundur case.
    """
    V1 = sample_relative(V1_NOM, V_PERT_REL)
    V2 = sample_relative(V2_NOM, V_PERT_REL)
    V3 = sample_relative(V3_NOM, V_PERT_REL)

    B12 = sample_relative(B12_NOM, B_PERT_REL)
    B13 = sample_relative(B13_NOM, B_PERT_REL)
    B23 = sample_relative(B23_NOM, B_PERT_REL)

    delta1 = sample_absolute(DELTA_STAR_NOM[0], DELTA_PERT_ABS)
    delta2 = sample_absolute(DELTA_STAR_NOM[1], DELTA_PERT_ABS)
    delta3 = 0.0

    m1 = sample_relative(M_NOM[0, 0], M_PERT_REL)
    m2 = sample_relative(M_NOM[1, 1], M_PERT_REL)
    m3 = sample_relative(M_NOM[2, 2], M_PERT_REL)

    d1 = sample_relative(D_NOM[0, 0], D_PERT_REL)
    d2 = sample_relative(D_NOM[1, 1], D_PERT_REL)
    d3 = sample_relative(D_NOM[2, 2], D_PERT_REL)

    case = {
        "V1": V1, "V2": V2, "V3": V3,
        "B12": B12, "B13": B13, "B23": B23,
        "delta_star": np.array([delta1, delta2, delta3], dtype=float),
        "M_np": np.diag([m1, m2, m3]),
        "D_np": np.diag([d1, d2, d3]),
    }

    if not is_physically_reasonable_case(case):
        return None

    return case

# ============================================================
# LMI solver
# ============================================================

def try_solve_problem(prob):
    # Try CLARABEL first if available
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        return
    except Exception:
        pass

    # Fallback to SCS
    prob.solve(
        solver=cp.SCS,
        verbose=False,
        max_iters=50000,
        eps=1e-6,
        acceleration_lookback=10
    )


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try_solve_problem(prob)
    except Exception:
        return None

    if prob.status != "optimal":
        return None

    if Q.value is None or k.value is None or h.value is None:
        return None

    Q_num = np.array(Q.value, dtype=float)
    k_num = np.array(k.value, dtype=float).reshape(-1)
    h_num = np.array(h.value, dtype=float).reshape(-1)

    q_eigs = np.linalg.eigvalsh(Q_num)
    if np.min(q_eigs) <= MIN_EIG_Q:
        return None

    if np.min(k_num) <= MIN_POSITIVE_VALUE:
        return None

    if np.min(h_num) <= MIN_POSITIVE_VALUE:
        return None

    return {
        "Q_num": Q_num,
        "k_num": k_num,
        "h_num": h_num,
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
# Build one sample
# ============================================================

def build_sample_line():
    try:
        case = sample_case()
        if case is None:
            return None

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

        # Compact form exactly following the paper / original 01.py
        b12 = B12 * V1 * V2
        b13 = B13 * V1 * V3
        b23 = B23 * V2 * V3
        Gamma_np = np.diag([b12, b13, b23])

        Z3 = np.zeros((3, 3))
        I3 = np.eye(3)

        A_np = np.block([
            [Z3, I3],
            [Z3, -Minv_np @ D_np],
        ])

        C_np = np.hstack([E_NP, np.zeros((3, 3))])

        B_np = np.vstack([
            np.zeros((3, 3)),
            Minv_np @ E_NP.T @ Gamma_np
        ])

        sol = solve_lmi(A_np, B_np, C_np)
        if sol is None:
            return None

        Q_num = sol["Q_num"]
        k_num = sol["k_num"]

        # symbolic model
        x = sp.symbols("x0:6")
        x_delta = sp.Matrix(x[:3])
        omega = sp.Matrix(x[3:])

        E = sp.Matrix(E_NP)
        Gamma = sp.Matrix(Gamma_np)
        delta_star = sp.Matrix(delta_star_np)

        M = sp.Matrix(M_np)
        D = sp.Matrix(D_np)
        Minv = M.inv()

        edge_angle = E * (x_delta + delta_star)
        edge_angle_star = E * delta_star

        # extra symbolic sanity check:
        # sampled equilibrium itself already satisfies |delta_ij*| < pi/2
        # which is stronger than the paper's |delta_ij + delta_ij*| < pi
        # at x = 0.

        F = sp.Matrix([
            sp.sin(edge_angle[i]) - sp.sin(edge_angle_star[i])
            for i in range(3)
        ])

        xdot_delta = omega
        xdot_omega = -(Minv * D) * omega - (Minv * E.T * Gamma) * F

        f_vec = sp.Matrix.vstack(xdot_delta, xdot_omega)
        f_list = [sp.expand(f_vec[i]) for i in range(6)]

        # Lyapunov function from paper Eq. (9)-style structure
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

        ok, unexpected = has_only_allowed_symbols(f_list + [V_expr])
        if not ok:
            return None

        if not is_good_numeric_expr(f_list + [V_expr]):
            return None

        orig_tokens = [expr_to_prefix_no_end(fi) for fi in f_list]
        dim_prefix = f"{num_token(STATE_DIM)} {SEP} "
        orig_joined = dim_prefix + f" {SEP} ".join(orig_tokens)

        v_tokens = expr_to_prefix_no_end(V_expr)

        line = "1| " + orig_joined + "\t" + v_tokens + "\n"
        return line

    except Exception as e:
        print("Sample generation error:", e)
        return None

# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_total):
    lines = []
    attempts = 0
    max_attempts = n_total * 100

    pbar = tqdm(total=n_total, desc="Generating Kundur samples")

    while len(lines) < n_total and attempts < max_attempts:
        attempts += 1

        line = build_sample_line()
        if line is not None:
            lines.append(line)
            pbar.update(1)

    pbar.close()

    print(f"\nAttempts: {attempts}")
    print(f"Accepted: {len(lines)}")
    if attempts > 0:
        print(f"Acceptance ratio: {len(lines) / attempts:.4f}")

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
# Main
# ============================================================

def main():
    print("Generating Kundur power system dataset")
    print(f"Target samples: {N_TOTAL}")
    print("Physical constraints enabled:")
    print("  1) equilibrium edge angles satisfy |delta_ij*| < pi")
    print("  2) stronger convex-region filter |delta_ij*| < pi/2")

    lines = generate_dataset(N_TOTAL)

    save_all(lines)

    if lines:
        print("\nExample sample:\n")
        print(lines[0][:800])

if __name__ == "__main__":
    main()
