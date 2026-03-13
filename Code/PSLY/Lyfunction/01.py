# ============================================================
# Kundur 9-bus (reduced 3-generator classical model)
# Paper:
#   Lyapunov Functions Family Approach to Transient Stability Assessment
#
# Output format (as you required):
#   1|  <Original function tokens> \t <Lyapunov function tokens> \n
#
# Where:
#   Original function = INT+ N <SPECIAL_3> f0 <SPECIAL_3> f1 <SPECIAL_3> ... <SPECIAL_3> f5
#   Lyapunov function = V
#
# Important:
#   - NO END token (no leaf terminator)
#   - <SPECIAL_3> is ONLY used between ODE expressions (f0..f5) inside "Original function"
#   - Between original-function and Lyapunov-function use TAB '\t'
#   - Each sample ends with '\n'
#
# Extra (added summary):
#   - Print two extra lines in summary:
#       Original function\n
#       Lyapunov function
#     and then print readable equations for checking.
# ============================================================

import numpy as np
import cvxpy as cp
import sympy as sp

# ============================================================
# 0) Kundur 3-generator reduced data (from the paper tables/text)
# ============================================================

V1, V2, V3 = 1.0566, 1.0502, 1.0170
B12, B13, B23 = 0.739, 1.0958, 1.245

# Line susceptance-like terms used in Gamma (paper compact form)
b12 = B12 * V1 * V2
b13 = B13 * V1 * V3
b23 = B23 * V2 * V3
Gamma_np = np.diag([b12, b13, b23])

# Incidence matrix for edges (1-2), (1-3), (2-3)
E_np = np.array([
    [ 1, -1,  0],
    [ 1,  0, -1],
    [ 0,  1, -1],
], dtype=float)

# Equilibrium generator angles (choose delta3* = 0 as reference)
delta_star_np = np.array([-0.1005, 0.0583, 0.0], dtype=float)

# Machine parameters (paper uses mk=2, dk=1)
M_np = 2.0 * np.eye(3)
D_np = 1.0 * np.eye(3)
Minv_np = np.linalg.inv(M_np)

# ============================================================
# 1) Build compact model matrices (paper compact form)
# ============================================================

Z3 = np.zeros((3, 3))
I3 = np.eye(3)

# A for x = [delta - delta* ; omega]
A_np = np.block([
    [Z3, I3],
    [Z3, -Minv_np @ D_np],
])

# C maps state to edge angle differences: Cx = E*(delta-delta*)
C_np = np.hstack([E_np, np.zeros((3, 3))])

# B maps nonlinear edge term F(Cx) into omega dynamics
B_np = np.vstack([
    np.zeros((3, 3)),
    Minv_np @ E_np.T @ Gamma_np
])

n = 6  # state dim
m = 3  # number of nonlinear edge terms

# ============================================================
# 2) Solve LMI (paper Eq.(8)) to obtain Q, k, h
# ============================================================

def solve_lmi():
    eps = 1e-6

    Q = cp.Variable((n, n), symmetric=True)
    k = cp.Variable(m, nonneg=True)
    h = cp.Variable(m, nonneg=True)

    K = cp.diag(k)
    H = cp.diag(h)

    A = A_np
    B = B_np
    C = C_np

    # R = Q B - C^T H - (K C A)^T
    R = Q @ B - C.T @ H - (K @ C @ A).T

    # LMI:
    # [A^T Q + Q A, R;
    #  R^T,       -2H] <= 0
    LMI = cp.bmat([
        [A.T @ Q + Q @ A, R],
        [R.T, -2 * H]
    ])

    constraints = [
        Q >> eps * np.eye(n),
        k >= eps,
        h >= eps,
        LMI << 0,
        cp.trace(Q) == 1.0    # fix scaling (avoid unbounded)
    ]

    # Objective: maximize sum(k) (helps avoid trivial solutions)
    prob = cp.Problem(cp.Maximize(cp.sum(k)), constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=200000, eps=1e-6)

    return prob.status, prob.value, Q.value, k.value, h.value


status, obj, Q_num, k_num, h_num = solve_lmi()

print("\n================ LMI Result ================")
print("status:", status)
print("objective:", obj)
print("Gamma diag:", np.diag(Gamma_np))
print("E*delta_star:", E_np @ delta_star_np)

if status not in ("optimal", "optimal_inaccurate"):
    raise RuntimeError("LMI solve failed (infeasible or solver error). Try another solver or tune eps/objective.")

# Save numerical solution (useful for debugging/reuse)
np.savez("lff_solution_kundur3.npz", Q=Q_num, k=k_num, h=h_num,
         E=E_np, Gamma=Gamma_np, delta_star=delta_star_np,
         A=A_np, B=B_np, C=C_np)
print("Saved: lff_solution_kundur3.npz")

# ============================================================
# 3) Build symbolic dynamics f(x) and Lyapunov V(x)
# ============================================================

# State symbols x0..x5
x = sp.symbols("x0:6")
x_delta = sp.Matrix(x[:3])  # delta - delta*
omega = sp.Matrix(x[3:])    # omega

E = sp.Matrix(E_np)
Gamma = sp.Matrix(Gamma_np)
delta_star = sp.Matrix(delta_star_np)

M = 2.0 * sp.eye(3)
D = 1.0 * sp.eye(3)
Minv = M.inv()

# Nonlinear edge term:
# F = sin(E*(delta* + x_delta)) - sin(E*delta*)
edge_angle = E * (x_delta + delta_star)
edge_angle_star = E * delta_star
F = sp.Matrix([sp.sin(edge_angle[i]) - sp.sin(edge_angle_star[i]) for i in range(3)])

# Dynamics:
# xdot_delta = omega
# xdot_omega = -M^{-1}D omega - M^{-1}E^T Gamma F
xdot_delta = omega
xdot_omega = -(Minv * D) * omega - (Minv * E.T * Gamma) * F

f_vec = sp.Matrix.vstack(xdot_delta, xdot_omega)

# Keep expressions compact-ish (avoid aggressive simplify explosion)
f_list = [f_vec[i] for i in range(6)]

# Lyapunov:
# V = 1/2 x^T Q x - sum_i k_i ( cos(delta_e_i) + delta_e_i * sin(delta*_i) )
Q = sp.Matrix(Q_num)
k = sp.Matrix(k_num)

y = E * x_delta
delta_e = edge_angle_star + y

V_quad = sp.Rational(1, 2) * (sp.Matrix(x).T * Q * sp.Matrix(x))[0]

V_nl = 0
for i in range(3):
    V_nl += k[i] * (sp.cos(delta_e[i]) + delta_e[i] * sp.sin(edge_angle_star[i]))

V_expr = (V_quad - V_nl)

# ============================================================
# 4) Prefix tokenization (STRICT BASE-1000 FORMAT)
# ============================================================

SEP = "<SPECIAL_3>"

def encode_base1000(n):
    """
    将整数转换为 base 1000 的 token 字符串。
    例如: 44808 -> '44 808'
    """
    if n == 0:
        return "0"
    chunks = []
    n = abs(int(n))
    while n > 0:
        chunks.append(str(n % 1000))
        n = n // 1000
    return " ".join(reversed(chunks))

def num_token(v):
    """
    将 Sympy 数字转换为符合模型词汇表的 Base 1000 和 科学计数法格式。
    """
    fv = float(v)
    
    # 1. 整数处理
    if abs(fv - round(fv)) < 1e-12:
        iv = int(round(fv))
        sign_str = "INT+" if iv >= 0 else "INT-"
        return f"{sign_str} {encode_base1000(iv)}"
        
    # 2. 浮点数处理 (科学计数法)
    if fv == 0:
        return "FLOAT+ 0 10^ INT+ 0"
        
    sign_str = "FLOAT+" if fv > 0 else "FLOAT-"
    fv = abs(fv)
    
    # 使用科学计数法，保留4位小数，共5位有效数字
    s = "{:.4e}".format(fv)
    mantissa, exponent = s.split('e')
    
    # 提取尾数数字部分并按 Base 1000 拆解
    mantissa_int = int(mantissa.replace('.', ''))
    mantissa_tokens = encode_base1000(mantissa_int)
    
    # 提取指数部分
    exp_val = int(exponent)
    exp_sign = "INT+" if exp_val >= 0 else "INT-"
    
    return f"{sign_str} {mantissa_tokens} 10^ {exp_sign} {abs(exp_val)}"

def expr_to_prefix_no_end(expr):
    """
    Convert a SymPy expression to prefix tokens WITHOUT leaf terminator.
    """
    def rec(e):
        if e.is_Number:
            # 保证拆分成单个 Token
            return num_token(e).split(' ')
            
        if e.is_Symbol:
            sym_name = str(e)
            mapping = {"delta_s": "x8"}
            return [mapping.get(sym_name, sym_name)]

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

    # 适度化简以维持结构稳定
    return " ".join(rec(sp.simplify(expr)))

# ============================================================
# 5) Build and Export Dataset Line
# ============================================================

# Build "Original function" tokens: f0 <SPECIAL_3> f1 ...
orig_tokens = [expr_to_prefix_no_end(fi) for fi in f_list]

# 核心修复 2：在系统方程最前面，加上该系统方程的数量（即维度），例如 "INT+ 6 <SPECIAL_3> "
# 解决 TypeError: 'float' object cannot be interpreted as an integer
dim_prefix = f"INT+ {encode_base1000(len(f_list))} {SEP} "
orig_joined = dim_prefix + f" {SEP} ".join(orig_tokens)

# Build Lyapunov tokens: V
v_tokens = expr_to_prefix_no_end(V_expr)

# Final dataset line: 1| <orig> \t <V> \n
line = "1|" + orig_joined + "\t" + v_tokens + "\n"

# Output file name
out_name = "PS_kundur3_lff"
with open(out_name, "w", encoding="utf-8") as f:
    f.write(line)

# ============================================================
# 6) Summary display
# ============================================================

print("\n================ Summary ================")
print("Original function")
print("Lyapunov function")

print("\nOriginal function (readable):")
for i, fi in enumerate(f_list):
    print(f"f{i} =", sp.simplify(fi))

print("\nLyapunov function (readable):")
print("V =", sp.simplify(V_expr))

print("========================================\n")

print(f"Wrote: {out_name}")
print("Done.")
