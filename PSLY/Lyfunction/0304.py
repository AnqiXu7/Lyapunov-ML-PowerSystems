# ============================================================
# 03+04: Anghel/Milano/Papachristodoulou (IEEE TCAS-I 2013)
# "Algorithmic Construction of Lyapunov Functions for Power System Stability Analysis"
#
# This script:
#   (1) hard-codes Model A and Model B original dynamics (shifted coordinates)
#   (2) hard-codes the paper-provided Lyapunov function V(x) for each model
#   (3) exports in BPoly-like "machine language" format:
#         1| <f1_prefix> <SPECIAL_3> <f2_prefix> <SPECIAL_3> <f3_prefix> <SPECIAL_3> <f4_prefix> \t <V_prefix>\n
#
# Notes:
#   - NO leaf/end token is used (NO extra <SPECIAL_3> inside expressions).
#   - Numbers are encoded using STRICT BASE-1000 and Scientific Notation.
#   - Variable mapping (paper -> dataset):
#        paper: x1, x2, x3, x4
#        here : x0, x1, x2, x3
#        mapping: x0=x1, x1=x2, x2=x3, x3=x4
# ============================================================

import sympy as sp

# ----------------------------
# 0) Symbols (dataset-style)
# ----------------------------
x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")  # x0..x3

SIN = sp.sin
COS = sp.cos

SEP = "<SPECIAL_3>"  # formula separator (between ODE components)

# ============================================================
# 1) Model 03: Model A (3-machine, no transfer conductances)
#    Original (shifted) dynamics in your screenshot:
#       x1dot = x2
#       x2dot = -sin(x1) - 0.5 sin(x1-x3) - 0.4 x2
#       x3dot = x4
#       x4dot = -0.5 sin(x3) - 0.5 sin(x3-x1) - 0.5 x4 + 0.05
#
#    Under our mapping: (paper x1,x2,x3,x4) -> (x0,x1,x2,x3)
# ============================================================
fA = [
    x1,
    -SIN(x0) - sp.Float("0.5") * SIN(x0 - x2) - sp.Float("0.4") * x1,
    x3,
    -sp.Float("0.5") * SIN(x2) - sp.Float("0.5") * SIN(x2 - x0) - sp.Float("0.5") * x3 + sp.Float("0.05"),
]

# Lyapunov function for Model A (paper gives V(x) in original phase space coordinates)
VA = (
    sp.Float("0.0932") * SIN(x0)
    - sp.Float("0.2920") * x3
    - sp.Float("25.3499") * COS(x0)
    - sp.Float("21.0067") * COS(x2)
    - sp.Float("0.0408") * x1
    - sp.Float("0.3359") * SIN(x2)
    - sp.Float("2.6408") * COS(x0) * COS(x2)
    + sp.Float("0.0165") * COS(x0) * SIN(x0)
    + sp.Float("0.1450") * COS(x0) * SIN(x2)
    - sp.Float("0.1098") * COS(x2) * SIN(x0)
    + sp.Float("0.1909") * COS(x2) * SIN(x2)
    - sp.Float("5.0017") * SIN(x0) * SIN(x2)
    - sp.Float("1.6016") * (COS(x0) ** 2)
    - sp.Float("1.1354") * (COS(x2) ** 2)
    + sp.Float("4.6283") * x1 * x3
    - sp.Float("0.02086") * x1 * COS(x0)
    + sp.Float("0.0616") * x1 * COS(x2)
    + sp.Float("0.0199") * x3 * COS(x0)
    + sp.Float("0.2721") * x3 * COS(x2)
    + sp.Float("3.5181") * x1 * SIN(x0)
    + sp.Float("1.52425") * x1 * SIN(x2)
    + sp.Float("0.6551") * x3 * SIN(x0)
    + sp.Float("5.2582") * x3 * SIN(x2)
    + sp.Float("11.0457") * (x1 ** 2)
    + sp.Float("12.8486") * (x3 ** 2)
    + sp.Float("51.7345")
)

# ============================================================
# 2) Model 04: Model B (2-machine vs infinite bus, with transfer conductances)
#    Original (shifted) dynamics in your screenshot:
#       x1dot = x2
#       x2dot = 33.5849 x1 - 1.8868 cos(x1-x3) - 5.2830 cos(x1)
#              - 16.9811 sin(x1-x3) - 59.6226 sin(x1) - 1.8868 x2
#       x3dot = x4
#       x4dot = 11.3924 sin(x1-x3) - 1.2658 cos(x1-x3) - 3.2278 cos(x3)
#              - 1.2658 x4 - 99.3671 sin(x3) + 48.4810
#
#    Under our mapping: (paper x1,x2,x3,x4) -> (x0,x1,x2,x3)
# ============================================================
fB = [
    x1,
    sp.Float("33.5849") * x0
    - sp.Float("1.8868") * COS(x0 - x2)
    - sp.Float("5.2830") * COS(x0)
    - sp.Float("16.9811") * SIN(x0 - x2)
    - sp.Float("59.6226") * SIN(x0)
    - sp.Float("1.8868") * x1,
    x3,
    sp.Float("11.3924") * SIN(x0 - x2)
    - sp.Float("1.2658") * COS(x0 - x2)
    - sp.Float("3.2278") * COS(x2)
    - sp.Float("1.2658") * x3
    - sp.Float("99.3671") * SIN(x2)
    + sp.Float("48.4810"),
]

# Lyapunov function for Model B (paper gives V(x) in original phase space coordinates)
VB = (
    sp.Float("1.2468") * COS(x0) * SIN(x0)
    - sp.Float("0.3646") * x3
    - sp.Float("18.7585") * COS(x0)
    - sp.Float("27.6219") * COS(x2)
    - sp.Float("6.9358") * SIN(x0)
    - sp.Float("4.1573") * SIN(x2)
    - sp.Float("7.2379") * COS(x0) * COS(x2)
    - sp.Float("0.3399") * x1
    + sp.Float("2.5142") * COS(x0) * SIN(x2)
    + sp.Float("5.6889") * COS(x2) * SIN(x0)
    + sp.Float("1.6431") * COS(x2) * SIN(x2)
    - sp.Float("2.5392") * SIN(x0) * SIN(x2)
    - sp.Float("11.3052") * (COS(x0) ** 2)
    - sp.Float("13.3274") * (COS(x2) ** 2)
    + sp.Float("0.0841") * x1 * x3
    + sp.Float("0.0939") * x1 * COS(x0)
    + sp.Float("0.2461") * x1 * COS(x2)
    + sp.Float("0.2212") * x3 * COS(x0)
    + sp.Float("0.1434") * x3 * COS(x2)
    + sp.Float("0.7038") * x1 * SIN(x0)
    - sp.Float("0.1629") * x1 * SIN(x2)
    + sp.Float("0.2459") * x3 * SIN(x0)
    + sp.Float("0.4671") * x3 * SIN(x2)
    + sp.Float("0.3647") * (x1 ** 2)
    + sp.Float("0.3158") * (x3 ** 2)
    + sp.Float("78.2509")
)

# ============================================================
# 3) Prefix tokenization (STRICT BASE-1000 FORMAT)
# ============================================================
def encode_base1000(n):

    if n == 0:
        return "0"
    chunks = []
    n = abs(int(n))
    while n > 0:
        chunks.append(str(n % 1000))
        n = n // 1000
    return " ".join(reversed(chunks))

def num_token(v) -> str:

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
    mantissa, exponent = s.split('e')

    mantissa_int = int(mantissa.replace('.', ''))
    mantissa_tokens = encode_base1000(mantissa_int)

    exp_val = int(exponent)
    exp_sign = "INT+" if exp_val >= 0 else "INT-"

    return f"{sign_str} {mantissa_tokens} 10^ {exp_sign} {abs(exp_val)}"

def expr_to_prefix(expr) -> str:
    def rec(e):
        e = sp.simplify(e)

        if e.is_Number:
            return num_token(e).split(' ')

        # sin/cos
        if e.func == sp.sin:
            return ["sin"] + rec(e.args[0])
        if e.func == sp.cos:
            return ["cos"] + rec(e.args[0])

        # add/mul/pow
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

        # fallback
        return [str(e)]

    return " ".join(rec(expr))

def build_line(f_list, V_expr) -> str:
    f_tokens = [expr_to_prefix(fi) for fi in f_list]

    dim_prefix = f"INT+ {encode_base1000(len(f_list))} {SEP} "
    original = dim_prefix + f" {SEP} ".join(f_tokens)

    Vtok = expr_to_prefix(V_expr)
    return f"1|{original}\t{Vtok}\n"

# ============================================================
# 4) Export + Summary
# ============================================================
dataset_out = "PS_Anghel_03_04"

lines = []
lines.append(build_line(fA, VA))
lines.append(build_line(fB, VB))

with open(dataset_out, "w", encoding="utf-8") as f:
    for ln in lines:
        f.write(ln)

print("==============================================")
print("Summary")
print("==============================================")

print("\n[Model 03] Original function:")
for i, fi in enumerate(fA, 1):
    print(f"  f{i} =", sp.simplify(fi))
print("\n[Model 03] Lyapunov function:")
print("  V =", sp.simplify(VA))

print("\n----------------------------------------------")

print("\n[Model 04] Original function:")
for i, fi in enumerate(fB, 1):
    print(f"  f{i} =", sp.simplify(fi))
print("\n[Model 04] Lyapunov function:")
print("  V =", sp.simplify(VB))

print("\n==============================================")
print("Wrote dataset file:")
print(" ", dataset_out)
print("Format per line: 1| INT+ 4 <SPECIAL_3> f0 <SPECIAL_3> ... \\t V \\n")
print("==============================================")
