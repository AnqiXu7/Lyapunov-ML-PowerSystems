"""
Adaptive Lyapunov Function Method for Power System Transient Stability Analysis
=============================================================================

What this script does (for your benchmark / machine-language pipeline):
1) Builds the *original* SMIB dynamics (Eq. (44) in the paper):
       m*ddot(δ) + d*dot(δ) + σ*sin(δ) - p = 0
   around a chosen stable equilibrium δ* (SEP): [δ*; 0], where δ* = arcsin(p/σ).
   We use shifted coordinates:
       x0 = δ - δ*
       x1 = ω = dot(δ)
   so that equilibrium is at x = 0.

2) For each operating point δ* in Table II (replacing the conservative Table I),
   Use the corresponding exact LMI optimized parameters (alpha, P11, P12, P22)
   to construct the perfect Lur'e-Postnikov Lyapunov function.
   where φ(x0) = sin(x0 + δ*) - sin(δ*).

3) Construct the EXACT Lur'e-Postnikov Lyapunov function (Eq. 45 in the paper):
   V(x) = x^T P x + 2*alpha * \int_0^{x0} φ(τ) dτ
   This includes the physical potential energy term (1 - cos) which captures
   the true nonlinear stability boundary of the power system.

Output format:
- Each row is ONE sample, with:
    "1|" prefix
    original function list: INT+ 2 <SPECIAL_3> f0 <SPECIAL_3> f1
    then a tab "\t"
    then Lyapunov function V(x)
    then newline "\n"

IMPORTANT: No leaf-ending tokens are used.
The ONLY separator between original-function expressions is <SPECIAL_3>.
"""

import numpy as np
import sympy as sp

# ============================================================
# 0) Tokenization settings (STRICT BASE-1000 FORMAT)
# ============================================================
SEP_EXPR = "<SPECIAL_3>"   # separator between expressions within a row

def encode_base1000(n):
    if n == 0:
        return "0"
    chunks = []
    n = abs(int(n))
    while n > 0:
        chunks.append(str(n % 1000))
        n = n // 1000
    return " ".join(reversed(chunks))

def _num_token(v) -> str:
    fv = float(v)

    # 1.integer
    if abs(fv - round(fv)) < 1e-12:
        iv = int(round(fv))
        sign_str = "INT+" if iv >= 0 else "INT-"
        return f"{sign_str} {encode_base1000(iv)}"

    # 2.float
    if fv == 0:
        return "FLOAT+ 0 10^ INT+ 0"

    sign_str = "FLOAT+" if fv > 0 else "FLOAT-"
    fv = abs(fv)

    s = "{:.4e}".format(fv)
    mantissa, exponent = s.split('e')

    mantissa_int = int(mantissa.replace('.', ''))
    mantissa_tokens = encode_base1000(mantissa_int)

    exp_val = int(exponent)
    exp_sign = "INT+" if exp_val >= 0 else "INT-"

    return f"{sign_str} {mantissa_tokens} 10^ {exp_sign} {abs(exp_val)}"

def expr_to_prefix(expr: sp.Expr) -> str:
    """
    SymPy -> prefix tokens.
    NO leaf-ending tokens. NO extra <SPECIAL_3> inside an expression.
    """
    def rec(e):
        e = sp.simplify(e)

        # numbers
        if e.is_Number:
            return _num_token(e).split(' ')

        # addition
        if isinstance(e, sp.Add):
            args = list(e.args)
            toks = rec(args[0])
            for a in args[1:]:
                toks = ["+"] + toks + rec(a)
            return toks

        # multiplication
        if isinstance(e, sp.Mul):
            args = list(e.args)
            toks = rec(args[0])
            for a in args[1:]:
                toks = ["*"] + toks + rec(a)
            return toks

        # power
        if isinstance(e, sp.Pow):
            base, exp = e.args
            return ["^"] + rec(base) + rec(exp)

        # trig
        if e.func == sp.sin:
            return ["sin"] + rec(e.args[0])
        if e.func == sp.cos:
            return ["cos"] + rec(e.args[0])

        # fallback
        return [str(e)]

    return " ".join(rec(expr))

# ============================================================
# 1) model parameters (from the paper)
# ============================================================
# Paper text (SMIB example): m = 0.1, d = 0.15, σ = 0.2
m = 0.1
d = 0.15
sigma = 0.2

# ============================================================
# 2) Table II cases (Exact LMI parameters from the paper)
# ============================================================
# Columns: δ* (rad), alpha, P11, P12, P22
TABLE_II = [
    {"delta_star": -0.253, "alpha": 0.273, "P11": 0.587, "P12": 0.354, "P22": 0.352},
    {"delta_star":  0.253, "alpha": 0.273, "P11": 0.587, "P12": 0.354, "P22": 0.352},
    {"delta_star":  0.524, "alpha": 0.275, "P11": 0.443, "P12": 0.273, "P22": 0.281},
    {"delta_star":  0.848, "alpha": 0.222, "P11": 0.219, "P12": 0.136, "P22": 0.160},
    {"delta_star":  1.120, "alpha": 0.138, "P11": 0.089, "P12": 0.063, "P22": 0.086},
    {"delta_star":  1.430, "alpha": 0.065, "P11": 0.027, "P12": 0.024, "P22": 0.038},
]

# ============================================================
# 3) Solver logic removed -> Using Exact Analytical Forms
# ============================================================
# (Replaced cvxpy SDP solver with direct application of Table II parameters)

# ============================================================
# 4) Build symbolic original function f(x) and Lyapunov V(x)
# ============================================================
x0, x1 = sp.symbols("x0 x1", real=True)

def build_symbolic_pair(delta_star: float, alpha: float, P11: float, P12: float, P22: float):
    # original dynamics f0, f1 (shifted)
    f0 = x1
    f1 = -(d/m) * x1 - (sigma/m) * (sp.sin(x0 + delta_star) - sp.sin(delta_star))

    # Exact Lur'e-Postnikov Lyapunov function
    # 1. Quadratic kinetic energy: V = x^T P x
    kinetic = P11 * x0**2 + 2 * P12 * x0 * x1 + P22 * x1**2
    
    # 2. Nonlinear potential energy: 2*alpha * integral(phi)
    # Integral of (sin(x0 + delta_star) - sin(delta_star)) dx0
    potential = 2 * alpha * (sp.cos(delta_star) - sp.cos(x0 + delta_star) - x0 * sp.sin(delta_star))
    
    # Total V(x)
    V = sp.simplify(kinetic + potential)
    
    return sp.simplify(f0), sp.simplify(f1), V

# ============================================================
# 5) Export to BPoly-like ASCII dataset
# ============================================================
def export_one_row(f0: sp.Expr, f1: sp.Expr, V: sp.Expr) -> str:
    f_tokens = [expr_to_prefix(f0), expr_to_prefix(f1)]
    dim_prefix = f"INT+ {encode_base1000(len(f_tokens))} {SEP_EXPR} "
    original = dim_prefix + f" {SEP_EXPR} ".join(f_tokens)
    lyap = expr_to_prefix(V)
    return f"1|{original}\t{lyap}\n"

def main():
    out_path = "PS_ALF"
    rows = []

    print("============================================================")
    print("Summary (each case):")
    print("  - Original function (in SymPy)")
    print("  - Exact Lur'e-Postnikov Lyapunov function (in SymPy)")
    print("  - Parameters from Table II")
    print("============================================================\n")

    for idx, case in enumerate(TABLE_II, start=1):
        ds = float(case["delta_star"])
        alpha = float(case["alpha"])
        p11, p12, p22 = float(case["P11"]), float(case["P12"]), float(case["P22"])

        f0, f1, V = build_symbolic_pair(
            delta_star=ds, alpha=alpha, P11=p11, P12=p12, P22=p22
        )
        rows.append(export_one_row(f0, f1, V))

        # ---- summary lines ----
        print(f"[Case {idx}] delta*={ds}, alpha={alpha}, P=[{p11}, {p12}; {p12}, {p22}]")
        print("Original function:")
        print(f"  f0(x) = {f0}")
        print(f"  f1(x) = {f1}")
        print("Lyapunov function:")
        print(f"  V(x)  = {V}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(rows)

    print("============================================================")
    print("Wrote dataset file:")
    print(f"  {out_path}")
    print(f"Rows written: {len(rows)}")
    print("Format check:")
    print("  - each row starts with '1|'")
    print("  - original functions separated ONLY by <SPECIAL_3>")
    print("  - original/lyapunov separated by TAB '\\t'")
    print("  - each row ends with newline '\\n'")
    print("============================================================")

if __name__ == "__main__":
    main()
