# ============================================================
# 06 Rational Lyapunov (Han, El-Guindy, Althoff, IEEE PESGM 2016)
# "Power systems transient stability analysis via optimal rational Lyapunov functions"
#
# System:
#   Double-machine versus infinite bus power system (4D), with transfer conductances.
#   Stable equilibrium point (SEP) in original coordinates:
#       (x1, x2, x3, x4) = (0.4680, 0, 0.4630, 0)
#
# Shifted coordinates (paper uses y):
#   y = (y1, y2, y3, y4)^T = (x1 - 0.4680, x2, x3 - 0.4630, x4)^T
#
# Dataset variable mapping:
#   x0 = y1
#   x1 = y2
#   x2 = y3
#   x3 = y4
#
# Output format:
#   1| <f0_prefix> <SPECIAL_3> <f1_prefix> <SPECIAL_3> <f2_prefix> <SPECIAL_3> <f3_prefix> \t <V_prefix> \n
#
# Important:
#   - NO END token (no leaf terminator)
#   - <SPECIAL_3> is ONLY used between ODE expressions (f0..f3) inside "Original function"
#   - Between original-function and Lyapunov-function use TAB '\t'
#   - Each sample ends with '\n'
#   - Numbers are output as FLOAT tokens
# ============================================================

import sympy as sp

# ============================================================
# 0) Tokenization settings (STRICT BASE-1000 FORMAT)
# ============================================================
SEP = "<SPECIAL_3>"

def encode_base1000(n):
    if n == 0:
        return "0"
    chunks = []
    n = abs(int(n))
    while n > 0:
        chunks.append(str(n % 1000))
        n = n // 1000
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
    mantissa, exponent = s.split('e')

    mantissa_int = int(mantissa.replace('.', ''))
    mantissa_tokens = encode_base1000(mantissa_int)

    exp_val = int(exponent)
    exp_sign = "INT+" if exp_val >= 0 else "INT-"

    return f"{sign_str} {mantissa_tokens} 10^ {exp_sign} {abs(exp_val)}"

def expr_to_prefix_no_end(expr: sp.Expr) -> str:
    def rec(e):
        e = sp.simplify(e)
        if e.is_Number:
            return num_token(e).split(' ')

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
            name = e.func.__name__
            return [name] + rec(e.args[0])

        return [str(e)]

    return " ".join(rec(expr))

# ============================================================
# 1) Variables (shifted coordinates y, but we name them x0..x3)
# ============================================================
# x0=x1-0.4680, x1=x2, x2=x3-0.4630, x3=x4
x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3", real=True)

# ============================================================
# 2) Original system (in shifted coordinates around SEP)
# ============================================================
theta1 = x0 + sp.Float("0.4680")
theta3 = x2 + sp.Float("0.4630")
theta13 = theta1 - theta3

f0 = x1
f1 = (
    sp.Float("33.5849")
    - sp.Float("1.8868") * sp.cos(theta13)
    - sp.Float("5.2830") * sp.cos(theta1)
    - sp.Float("59.6226") * sp.sin(theta1)
    - sp.Float("16.9811") * sp.sin(theta13)
    - sp.Float("1.8868") * x1
)
f2 = x3
f3 = (
    sp.Float("48.4810")
    + sp.Float("11.3924") * sp.sin(theta13)
    - sp.Float("3.2278") * sp.cos(theta3)
    - sp.Float("99.3761") * sp.sin(theta3)
    - sp.Float("1.2658") * sp.cos(theta13)
    - sp.Float("1.2658") * x3
)

f_list = [f0, f1, f2, f3]

# ============================================================
# 3) Rational Lyapunov function V(y)
# ============================================================
y1, y2, y3, y4 = x0, x1, x2, x3

num = (
    y1**2 + y2**2 + y3**2 + y4**2
    + y1**4
    - y1**2 * y3**2
    + y3**4
)

den = (
    sp.Float("1.0")
    + sp.Float("2.0") * y1
    + y2
    - sp.Float("2.0") * y3
    + sp.Float("8.0") * y1**2
    + sp.Float("4.0") * y2**2
    + sp.Float("4.0") * y3**2
)

V_expr = sp.simplify(num / den)

# ============================================================
# 4) Export line
# ============================================================
def export_line():
    f_tokens = [expr_to_prefix_no_end(fi) for fi in f_list]

    dim_prefix = f"INT+ {encode_base1000(len(f_list))} {SEP} "
    original = dim_prefix + f" {SEP} ".join(f_tokens)

    v_tokens = expr_to_prefix_no_end(V_expr)
    return "1|" + original + "\t" + v_tokens + "\n"

line = export_line()

out_name = "PS_rational_double_machine"
with open(out_name, "w", encoding="utf-8") as f:
    f.write(line)

# ============================================================
# 5) Summary display
# ============================================================
print("\n================ Summary ================")
print("Original function")
print("Lyapunov function")

print("\nOriginal function (readable):")
for i, fi in enumerate(f_list):
    print(f"f{i} =", sp.simplify(fi))

print("\nLyapunov function (readable):")
print("V =", V_expr)

print("\n========================================")
print(f"Wrote: {out_name}")
print("Done.")
