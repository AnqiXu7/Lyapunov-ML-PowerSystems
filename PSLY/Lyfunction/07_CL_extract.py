# ============================================================
# 07_control_loop_extract.py
# 3D VSG dataset with additional first-order control loop
# EXPANSION MODE
#
# Generate symbolic datasets for learning Lyapunov functions
# from a VSG system with an extra control-loop state.
#
# Output format:
#
#   1| INT+ 3 <SPECIAL_3> f0 <SPECIAL_3> f1 <SPECIAL_3> f2 \t V
#
# ============================================================

import random
import sympy as sp
from tqdm import tqdm

# ============================================================
# Global settings
# ============================================================

SEP = "<SPECIAL_3>"
STATE_DIM = 3

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

x0, x1, x2 = sp.symbols("x0 x1 x2", real=True)

# ============================================================
# System parameters
# ============================================================

J, D = sp.symbols("J D", positive=True)
T0 = sp.symbols("T0")
E, Vg = sp.symbols("E Vg", positive=True)
B, G = sp.symbols("B G")
omega0 = sp.symbols("omega0", positive=True)
delta_s = sp.symbols("delta_s")
lam = sp.symbols("lam", positive=True)

kc, a, b, mu = sp.symbols("kc a b mu", positive=True)

# ============================================================
# System dynamics
# ============================================================

f0_sym = x1

f1_sym = (
    T0
    - (E * Vg * B / omega0) * sp.sin(delta_s + x0)
    - (E * Vg * G / omega0) * sp.cos(delta_s + x0)
    - D * x1
    - kc * x2
) / J

f2_sym = -a * x2 + b * x1

f_list_sym = [f0_sym, f1_sym, f2_sym]

# ============================================================
# Lyapunov function
# ============================================================

V_sym = (
    sp.Rational(1,2)*J*x1**2
    - T0*x0
    + (E*Vg*B/omega0)*(sp.cos(delta_s+x0)-sp.cos(delta_s))
    - (E*Vg*G/omega0)*(sp.sin(delta_s+x0)-sp.sin(delta_s))
    + D*lam*x0*x1
    + (D**2/(2*J))*lam*x0**2
    + sp.Rational(1,2)*mu*x2**2
    + lam*kc*x0*x2
)

# ============================================================
# Parameter sampling
# ============================================================

def sample_parameters():

    omega0_num = 314.159

    E_num = random.uniform(280,340)
    Vg_num = random.uniform(280,340)

    R_num = random.uniform(0.05,0.2)
    X_num = random.uniform(0.15,0.4)

    denom = R_num**2 + X_num**2

    G_num = R_num/denom
    B_num = -X_num/denom

    J_num = random.uniform(3,8)
    D_num = random.uniform(2,12)
    lam_num = random.uniform(0.2,1)

    delta_s_num = random.uniform(0.2,0.8)

    P0_num = random.uniform(10000,30000)
    T0_num = P0_num/omega0_num

    kc_num = random.uniform(0.5,8)
    a_num = random.uniform(0.5,5)
    b_num = random.uniform(0.5,5)
    mu_num = random.uniform(0.5,5)

    return {
        J:J_num, D:D_num, T0:T0_num,
        E:E_num, Vg:Vg_num,
        omega0:omega0_num,
        G:G_num, B:B_num,
        lam:lam_num,
        delta_s:delta_s_num,
        kc:kc_num, a:a_num, b:b_num, mu:mu_num
    }

# ============================================================
# Build sample
# ============================================================

def build_sample_line(subs):

    f_list = [sp.simplify(fi.subs(subs)) for fi in f_list_sym]
    V_expr = sp.simplify(V_sym.subs(subs))

    for expr in f_list+[V_expr]:
        s=str(expr)
        if "nan" in s or "oo" in s or "zoo" in s:
            return None

    original_tokens=[expr_to_prefix_no_end(fi) for fi in f_list]

    dim_prefix=f"INT+ {encode_base1000(STATE_DIM)} {SEP} "

    original_joined=dim_prefix+f" {SEP} ".join(original_tokens)

    V_tokens=expr_to_prefix_no_end(V_expr)

    line="1|"+original_joined+"\t"+V_tokens+"\n"

    return line

# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_total):

    lines=[]
    attempts=0

    pbar=tqdm(total=n_total,desc="Generating VSG control-loop samples")

    while len(lines)<n_total:

        attempts+=1

        subs=sample_parameters()

        line=build_sample_line(subs)

        if line is not None:
            lines.append(line)
            pbar.update(1)

    pbar.close()

    print("Total attempts:",attempts)

    return lines

# ============================================================
# Save dataset
# ============================================================

def save_all(lines):

    with open("PS_VSG_CTRL_all","w",encoding="utf-8") as f:
        f.writelines(lines)

    print("\nSaved dataset:")
    print("PS_VSG_CTRL_all :",len(lines))

# ============================================================
# Main
# ============================================================

def main():

    print("Generating VSG control-loop dataset")
    print("Target samples:",N_TOTAL)

    lines=generate_dataset(N_TOTAL)

    save_all(lines)

    print("\nDone.")

if __name__=="__main__":
    main()
