# ============================================================
# Classical 2-bus system (paper Section V-A)
# Lyapunov Functions Family Approach to Transient Stability Assessment
#
# Dataset format:
#   1| <Original function tokens> \t <Lyapunov function tokens> \n
#
# Original function:
#   INT+ 2 <SPECIAL_3> f0 <SPECIAL_3> f1
#
# ============================================================

import sympy as sp

# ============================================================
# 0) System definition
# ============================================================

# State variables
x0, x1 = sp.symbols("x0 x1")

# equilibrium
delta_star = sp.Float("0.5235987755982988")
cos_delta_star = sp.Float("0.8660254037844386")
sin_delta_star = sp.Float("0.5")

# dynamics
f0 = x1
f1 = -x1 - sp.Float("0.8") * sp.sin(x0 + delta_star) + sp.Float("0.4")

f_list = [f0, f1]

# Lyapunov function
V_expr = (
    sp.Float("0.5") * x1**2 +
    sp.Float("0.8") *
    (cos_delta_star - sp.cos(x0 + delta_star) - x0 * sin_delta_star)
)

# ============================================================
# Tokenization
# ============================================================

SEP = "<SPECIAL_3>"

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
            base, exp = e.args
            return ["^"] + rec(base) + rec(exp)

        if e.func == sp.sin:
            return ["sin"] + rec(e.args[0])

        if e.func == sp.cos:
            return ["cos"] + rec(e.args[0])

        return [str(e)]

    return " ".join(rec(sp.simplify(expr)))


# ============================================================
# Safety check
# ============================================================

allowed_symbols = {"x0", "x1"}

expr_symbols = set()
for expr in f_list + [V_expr]:
    expr_symbols |= {str(s) for s in expr.free_symbols}

unexpected = sorted(expr_symbols - allowed_symbols)

if unexpected:
    raise ValueError(f"Unexpected symbols found: {unexpected}")


# ============================================================
# Build dataset line
# ============================================================

orig_tokens = [expr_to_prefix_no_end(fi) for fi in f_list]

dim_prefix = f"INT+ {encode_base1000(len(f_list))} {SEP} "
orig_joined = dim_prefix + f" {SEP} ".join(orig_tokens)

v_tokens = expr_to_prefix_no_end(V_expr)

line = "1|" + orig_joined + "\t" + v_tokens + "\n"

out_name = "PS_2bus_energy"

with open(out_name, "w", encoding="utf-8") as f:
    f.write(line)


# ============================================================
# Summary
# ============================================================

print("\n================ Summary ================")

print("\nOriginal function:")
for i, fi in enumerate(f_list):
    print(f"f{i} =", sp.simplify(fi))

print("\nLyapunov function:")
print("V =", sp.simplify(V_expr))

print("========================================")

print(f"Wrote dataset to: {out_name}")
print("Done.")
