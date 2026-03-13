# ============================================================
# function 07: Transient Angle Stability of Virtual Synchronous Generators
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

def expr_to_prefix_no_end(expr):
    def rec(e):
        if e.is_Number:
            return num_token(e).split(' ')

        if e.is_Symbol:
            sym_name = str(e)
            return [sym_name]

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
# 1) Define states
# ============================================================
x0, x1 = sp.symbols("x0 x1", real=True)

x_angle = x0
x_speed = x1

# ============================================================
# 2) Define parameters
# ============================================================
J, D = sp.symbols("J D", positive=True, real=True)
T0 = sp.symbols("T0", real=True)
E, Vg = sp.symbols("E Vg", positive=True, real=True)
B, G = sp.symbols("B G", real=True)
omega0 = sp.symbols("omega0", positive=True, real=True)
delta_s = sp.symbols("delta_s", real=True)
lam = sp.symbols("lam", positive=True, real=True)

# ============================================================
# 3) Original function
# ============================================================
f0 = x_speed
f1 = (
    T0
    - (E * Vg * B / omega0) * sp.sin(delta_s + x_angle)
    - (E * Vg * G / omega0) * sp.cos(delta_s + x_angle)
    - D * x_speed
) / J

f_list = [f0, f1]

# ============================================================
# 4) Lyapunov function
# ============================================================
V = (
    sp.Rational(1, 2) * J * x_speed**2
    - T0 * x_angle
    + (E * Vg * B / omega0) * (sp.cos(delta_s + x_angle) - sp.cos(delta_s))
    - (E * Vg * G / omega0) * (sp.sin(delta_s + x_angle) - sp.sin(delta_s))
    + D * lam * x_angle * x_speed
    + (D**2 / (2 * J)) * lam * x_angle**2
)

# ============================================================
# 5) Numeric parameter substitution
# ============================================================
omega0_num = 314.159
E0_num = 311.0
Vg_num = 311.0
P0_num = 20000.0
J_num = 5.0224
D_num = 8.0
T0_num = P0_num / omega0_num
lam_num = 0.5

# line / network equivalent
R_num = 0.05 + 0.06
X_num = 0.2 + 0.03
G_num = R_num / (R_num**2 + X_num**2)
B_num = -X_num / (R_num**2 + X_num**2)

delta_s_num = 0.5

subs_numeric = {
    J: J_num,
    D: D_num,
    T0: T0_num,
    E: E0_num,
    Vg: Vg_num,
    omega0: omega0_num,
    G: G_num,
    B: B_num,
    lam: lam_num,
    delta_s: delta_s_num,
}

f_list_num = [sp.simplify(fi.subs(subs_numeric)) for fi in f_list]
V_num = sp.simplify(V.subs(subs_numeric))

# ============================================================
# 6) Safety check
# ============================================================
allowed_symbols = {"x0", "x1"}
expr_symbols = set()

for expr in f_list_num + [V_num]:
    expr_symbols |= {str(s) for s in expr.free_symbols}

unexpected = sorted(expr_symbols - allowed_symbols)
if unexpected:
    raise ValueError(f"Unexpected symbols found in expressions: {unexpected}")

# ============================================================
# 7) Export NUMERIC dataset
# ============================================================
def export_line():
    original_tokens = [expr_to_prefix_no_end(fi) for fi in f_list_num]

    dim_prefix = f"INT+ {encode_base1000(len(f_list_num))} {SEP} "
    original_joined = dim_prefix + f" {SEP} ".join(original_tokens)

    V_tokens = expr_to_prefix_no_end(V_num)

    return "1|" + original_joined + "\t" + V_tokens + "\n"

out_name = "PS_VSG"

with open(out_name, "w", encoding="utf-8") as f:
    f.write(export_line())

# ============================================================
# 8) Print summary
# ============================================================
print("Original function:")
for i, fi in enumerate(f_list_num):
    print(f"f{i} =", fi)

print("\nLyapunov function:")
print("V =", V_num)

print(f"\nWrote dataset file: {out_name}")
print("Done.")
