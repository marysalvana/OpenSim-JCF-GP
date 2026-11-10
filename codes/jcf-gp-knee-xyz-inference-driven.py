"""
Adaptive inference-driven block mixed-precision Cholesky factorization.

Benchmark on the trivariate separable space-time Matern emulator of the three
knee joint contact force components (the model, data, likelihood and
prediction code of jcf-gp-knee-xyz-global-budget.py, unchanged) of three rules
for choosing the storage precision of the off-diagonal blocks of the Cholesky
factor, at fixed covariance parameters:

  Method A  the per-block rule of Salvana (2026),
                ||fl_p(A_ik) - A_ik||_F ||L_kk||_2 <= tau m / n_B
  Method B  the global numerical error budget 2 sum ||E_ik L_kk^T||_F^2 <= tau^2 m^2,
            spent by the paced online rule (B-paced) and, where feasible, by the
            offline plan from recorded costs (B-optimal)
  Method C  Algorithm 2, the inference-driven rule: the same global numerical
            budget, paced, plus a second paced budget on a statistical
            sensitivity score computed from the observed data
  Method D  every block in fp64, the reference for likelihood accuracy

All arithmetic is fp64 and the diagonal factor blocks are stored in fp64; the
formats bf16, fp32 and fp64 are storage formats only.

Algorithm 2 in brief. The block forward substitution of the observed residuals
y is carried along the factorization: at block column k, z_k = L_kk^{-1} r_k
with r_k = y_k - sum_{j<k} L~_kj z_j, and after block (i, k) is stored,
r_i <- r_i - L~_ik z_k. Rounding A_ik to format p, E_ik,p = fl_p(A_ik) - A_ik,
perturbs the residual update of row block i by e_ik,p = E_ik,p z_k. With v_i
the diagonal of the current Schur complement of row block i, initialized to
diag(Sigma_ii) and reduced by the row sums of squares of every stored block of
that row, the inference-sensitivity score is
    s_ik,p = sum_a ( |r0_ia e_ik,p,a| + e_ik,p,a^2 / 2 ) / v_ia,     r0_i = r_i - A_ik z_k,
and the numerical contribution is d_ik,p^2 = 2 ||E_ik,p L_kk^T||_F^2. Two
budgets, B_N = tau^2 m^2 and B_S = epsilon, are spent at the paced rate: with q
off-diagonal blocks left, the allowances are b_N = (B_N - used_N) / q and
b_S = (B_S - used_S) / q, the lowest format with d^2 <= b_N and s <= b_S is
stored, fp64 if none qualifies, and the budgets are updated. Unused budget
rolls over.

Derivation of the score. Conditionally on the blocks before it, the
contribution of row block i to the Gaussian quadratic form y^T Sigma^{-1} y is
r_i^T S_i^{-1} r_i with S_i the Schur complement of row block i and r_i the
residual after the earlier blocks. Replacing S_i by its diagonal v_i gives the
approximation Q_i = sum_a r_ia^2 / v_ia. Storing block (i, k) in format p
changes the residual update from A_ik z_k to A_ik z_k + E_ik,p z_k, so the
residual after the update is r0_i - e_ik,p instead of r0_i, and the half
quadratic form changes by
    sum_a [ (r0_ia - e_a)^2 - r0_ia^2 ] / (2 v_ia) = sum_a ( -r0_ia e_a + e_a^2 / 2 ) / v_ia,
whose absolute value is at most s_ik,p by the triangle inequality. The score is
therefore an upper bound on the change that rounding one block causes in the
diagonal conditional-residual approximation of the half quadratic form, in
units of the standardized residuals, where the profiled variance is near one.
It is a sensitivity proxy and not a bound on the full likelihood: the later
residual updates, the later diagonal factorizations and the interactions of
rounded blocks are not in it, and the diagonal of the Schur complement replaces
the full conditional covariance. Nonpositive or missing entries of v_i are
replaced by the lower bound m on the smallest eigenvalue, which can only
increase the score.

Guarantee. Every variant of Methods B and C accepts a block only if the running
numerical total stays within tau^2 m^2, so, under exact fp64 arithmetic and
storage, the matrix of the blocks stored so far is always within tau m of
Sigma in Frobenius norm, the induction of Propositions 1 and 2 of the paper
applies unchanged, every diagonal Cholesky exists, and the completed factor
satisfies ||L~ L~^T - Sigma||_F <= tau m < lambda_min(Sigma), so L~ L~^T is
positive definite by Weyl's inequality. The statistical budget only removes
candidates, so it cannot break the guarantee. The implementation runs in
finite-precision fp64, whose rounding is not covered by the bound. The paced
two-budget rule is an online heuristic, not an optimal allocation, and since it
adds a restriction to the numerical budget it need not store less than the
numerical-only rules; whether it buys likelihood accuracy per byte is what the
epsilon sweep measures.

Usage
  python3 codes/jcf-gp-knee-xyz-inference-driven.py                                  # 24 strides, N = 7,272
  python3 codes/jcf-gp-knee-xyz-inference-driven.py --epsilon 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,inf
  python3 codes/jcf-gp-knee-xyz-inference-driven.py --strides 48 --no-reference --no-dense
  python3 codes/jcf-gp-knee-xyz-inference-driven.py --fit-each                       # also estimate the parameters under each method
Run from the project root; writes CSV tables and PNG figures to --out.
"""

import argparse
import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.optimize import minimize

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------- arguments

parser = argparse.ArgumentParser(description="Inference-driven precision assignment for the mixed-precision Cholesky factorization of the trivariate space-time knee JCF component model, against the per-block and the global numerical budgets.")
parser.add_argument("--joint", default="knee", choices=["hip", "knee", "ankle"], help="joint whose force components are modeled (default knee)")
parser.add_argument("--strides", type=int, default=24, help="number of training strides, evenly spaced over the 480 (default 24, N = 7,272)")
parser.add_argument("--block-size", type=int, default=303, help="points per block of the factorization (default 303, three strides of one component)")
parser.add_argument("--tau", type=float, default=0.99, help="error-budget parameter, 0 < tau < 1 (default 0.99)")
parser.add_argument("--epsilon", default="1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,inf", help="comma-separated statistical sensitivity tolerances for Method C; inf is the inactive constraint (default 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,inf)")
parser.add_argument("--epsilon-main", type=float, default=1e-5, help="the tolerance shown in the precision maps and used by --fit-each (default 1e-5)")
parser.add_argument("--plan-fraction", type=float, default=0.9, help="fraction of the numerical budget the offline plan of B-optimal may use (default 0.9)")
parser.add_argument("--theta", default=None, help="JSON checkpoint with the fitted parameters (default: results/global_budget_spacetime/spacetime_checkpoint_<joint>_xyz_<strides>strides.json)")
parser.add_argument("--max-iterations", type=int, default=20, help="L-BFGS-B iterations when parameters have to be fitted (default 20)")
parser.add_argument("--fit-each", action="store_true", help="also estimate the parameters under Method A, B-paced and Method C at --epsilon-main, and report the estimates side by side")
parser.add_argument("--test-strides", type=int, default=20, help="number of test strides used for the prediction check, 0 to skip (default 20)")
parser.add_argument("--no-reference", action="store_true", help="skip the all-fp64 reference (Method D)")
parser.add_argument("--no-optimal", action="store_true", help="skip the offline plan (B-optimal)")
parser.add_argument("--no-dense", action="store_true", help="skip the dense verification of the tracked error")
parser.add_argument("--data", default="data", help="folder with the train and test files (default data)")
parser.add_argument("--out", default="results/inference_driven", help="folder for the tables and figures (default results/inference_driven)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
EPSILONS = [float(e) for e in args.epsilon.split(",")]
if args.epsilon_main not in EPSILONS:
    EPSILONS.append(args.epsilon_main)
EPSILONS = sorted(set(EPSILONS))

# ----------------------------------------------------------------------------- setup

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.float64)
print(f"{platform.machine()}, {torch.get_num_threads()} CPU threads, torch {torch.__version__}")

# ----------------------------------------------------------------------------- data

x_train = pd.read_csv(os.path.join(args.data, "x-train.csv"))
y_train = pd.read_csv(os.path.join(args.data, "y-train.csv"))
x_test = pd.read_csv(os.path.join(args.data, "x-test.csv")).iloc[:max(args.test_strides, 1)].reset_index(drop=True)
y_test = pd.read_csv(os.path.join(args.data, "y-test.csv")).iloc[:max(args.test_strides, 1)].reset_index(drop=True)
n_all = len(x_train)
keep = np.unique(np.linspace(0, n_all - 1, min(args.strides, n_all)).round().astype(int))
x_train, y_train = x_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(drop=True)

GRID = np.arange(101)
COMPONENTS = ["fx", "fy", "fz"]
INPUT_CURVES = ["grf_vertical_BW", "grf_anteroposterior_BW", "grf_mediolateral_BW",
                "hip_flexion_deg", "hip_adduction_deg", "hip_rotation_deg", "knee_flexion_deg", "ankle_angle_deg"]


def curve(df, name):
    return df[[f"{name}_t{t:03d}" for t in GRID]].to_numpy()


def build_features(x, x_ref):
    cols = []
    for name in INPUT_CURVES:
        C, C_ref = curve(x, name), curve(x_ref, name)
        cols.extend(((C - C_ref.mean(axis=0)) / C_ref.std()).T)
    for name in ["belt_speed_mps", "mass_kg", "height_m"]:
        cols.append((x[name].to_numpy() - x_ref[name].mean()) / x_ref[name].std())
    return np.array(cols).T


F_train = build_features(x_train, x_train)
F_test = build_features(x_test, x_train)
Y_train = np.stack([curve(y_train, f"{args.joint}_jcf_{c}_BW") for c in COMPONENTS])     # (3 components, strides, 101 gait points), in BW
Y_test = np.stack([curve(y_test, f"{args.joint}_jcf_{c}_BW") for c in COMPONENTS])
mean_t = Y_train.mean(axis=1)                                                          # mean curve of each component
scale = Y_train.std(axis=(1, 2))                                                       # s_j, in BW
Z_train = (Y_train - mean_t[:, None, :]) / scale[:, None, None]                        # standardized residuals

# ----------------------------------------------------------------------------- trivariate separable space-time Matern model (unchanged)

def matern(d):
    s = 5.0**0.5 * d
    return (1.0 + s + s * s / 3.0) * torch.exp(-s)


def correlation_matrix(raw):
    # 3 x 3 correlation matrix from three unconstrained numbers, through the angles of its Cholesky factor
    a = (np.pi / 2) * np.tanh(np.asarray(raw, dtype=float))
    L = np.array([[1.0, 0.0, 0.0],
                  [np.sin(a[0]), np.cos(a[0]), 0.0],
                  [np.sin(a[1]), np.cos(a[1]) * np.sin(a[2]), np.cos(a[1]) * np.cos(a[2])]])
    return L @ L.T


def pair_ranges(l):
    # 1 / l_jk and gamma_jk for every pair of components
    a2 = 1.0 / l**2
    inverse = np.sqrt((a2[:, None] + a2[None, :]) / 2)
    return inverse, (np.sqrt(a2[:, None] * a2[None, :]) / inverse**2) ** 2.5


class SpaceTimeTrivariateMatern:
    def __init__(self, F, Z, t, block_size):
        self.n, self.T, self.N = Z.shape[1], Z.shape[2], Z.size
        Ft = torch.tensor(F, dtype=torch.float64, device=DEVICE)
        self.F = Ft
        self.H = torch.cdist(Ft, Ft, compute_mode="donot_use_mm_for_euclid_dist") / F.shape[1]**0.5      # h between strides
        self.t = torch.tensor(t, dtype=torch.float64, device=DEVICE)
        self.y = torch.tensor(Z.reshape(-1), dtype=torch.float64, device=DEVICE)                          # stacked residuals
        index = torch.arange(self.N, device=DEVICE)
        self.component, self.stride, self.time = index // (self.n * self.T), (index // self.T) % self.n, index % self.T
        self.blocks = [slice(a, min(a + block_size, self.N)) for a in range(0, self.N, block_size)]

    def set_parameters(self, theta):
        # theta = log input ranges (3), log temporal ranges (3), correlation angles (3), log variance ratios of fy and fz (2), log relative nugget effects (3)
        l, lt = np.exp(theta[0:3]), np.exp(theta[3:6])
        beta = correlation_matrix(theta[6:9])
        v = np.concatenate([[1.0], np.exp(theta[9:11])])
        eta = np.exp(theta[11:14])
        inverse_l, gamma = pair_ranges(l)
        inverse_lt, gamma_t = pair_ranges(lt)
        rho = beta * gamma * gamma_t                                               # cross-correlations at the same stride and gait point
        as_tensor = lambda a: torch.tensor(a, dtype=torch.float64, device=DEVICE)
        self.amplitude = as_tensor(rho * np.sqrt(np.outer(v, v)))
        self.inverse_l, self.inverse_lt = as_tensor(inverse_l), as_tensor(inverse_lt)
        self.nugget = as_tensor(eta)[self.component]                               # relative nugget effect of every point
        self.m = float(eta.min())                                                  # lower bound on the smallest eigenvalue
        self.parameters = dict(l=l, lt=lt, v=v, eta=eta, rho=rho)

    def block(self, rows, cols):
        # block of the covariance matrix between two sets of points, without the nugget effect
        pair = (self.component[rows][:, None], self.component[cols][None, :])
        h = self.H[self.stride[rows]][:, self.stride[cols]]
        u = (self.t[self.time[rows]][:, None] - self.t[self.time[cols]][None, :]).abs()
        return self.amplitude[pair] * matern(h * self.inverse_l[pair]) * matern(u * self.inverse_lt[pair])

    def cross(self, F_new, rows_train):
        # covariance between every (component, new stride, gait point) and the training points in rows_train, without the nugget effect
        Fn = torch.tensor(F_new, dtype=torch.float64, device=DEVICE)
        h_new = torch.cdist(Fn, self.F, compute_mode="donot_use_mm_for_euclid_dist") / self.F.shape[1]**0.5    # (new strides, training strides)
        n_new = Fn.shape[0]
        idx = torch.arange(3 * n_new * self.T, device=DEVICE)
        c, s, t = idx // (n_new * self.T), (idx // self.T) % n_new, idx % self.T
        pair = (c[:, None], self.component[rows_train][None, :])
        h = h_new[s][:, self.stride[rows_train]]
        u = (self.t[t][:, None] - self.t[self.time[rows_train]][None, :]).abs()
        return self.amplitude[pair] * matern(h * self.inverse_l[pair]) * matern(u * self.inverse_lt[pair])

# ----------------------------------------------------------------------------- storage formats, block costs and the precision rules

TAU = args.tau
CANDIDATES = {"bf16": torch.bfloat16, "fp32": torch.float32}      # lowest precision first
LARGEST = {"bf16": 3.39e38, "fp32": 3.40e38}                       # largest finite number of each format
FORMAT_NAMES = ["bf16", "fp32", "fp64"]
BYTES = {"bf16": 2, "fp32": 4, "fp64": 8}


def block_costs(A, Lkk):
    """Conversion error of A in every candidate format: the error matrix E_p, its Frobenius norm (per-block rule), and the
    realized contribution d^2 = 2 ||E_p L_kk^T||_F^2 to the squared Frobenius norm of the covariance perturbation (global
    budget). Overflowing formats are infeasible. fp64 is the reference and costs nothing."""
    largest = float(A.abs().max())
    E, err, d2, finite = {}, {}, {}, {}
    for name, dtype in CANDIDATES.items():
        Ep = A.to(dtype).to(torch.float64) - A
        finite[name] = largest <= LARGEST[name] and bool(torch.isfinite(Ep).all())
        E[name] = Ep if finite[name] else None
        err[name] = float(torch.linalg.norm(Ep)) if finite[name] else np.inf
        d2[name] = 2.0 * float(torch.linalg.norm(Ep @ Lkk.T))**2 if finite[name] else np.inf
    E["fp64"], err["fp64"], d2["fp64"], finite["fp64"] = None, 0.0, 0.0, True
    return E, err, d2, finite


def sensitivity(E, z_k, r0, v_i, m):
    """Inference-sensitivity score s = sum_a (|r0_a e_a| + e_a^2 / 2) / v_a with e = E z_k, the perturbation of the residual
    update of row block i by the rounding of block (i, k). Entries of v_i that are not positive are replaced by m."""
    if E is None:
        return 0.0, 0.0
    e = E @ z_k
    v = torch.where(torch.isfinite(v_i) & (v_i > 0), v_i, torch.full_like(v_i, m))
    return float((((r0 * e).abs() + 0.5 * e * e) / v).sum()), float(e.norm())


class PerBlockRule:
    """Method A: the original rule, ||fl_p(A) - A||_F ||L_kk||_2 <= tau m / n_B."""
    name = "A: per-block budget"

    def __init__(self, m, nB, n_blocks, **kw):
        self.b = TAU * m / nB

    def choose(self, info):
        for name in CANDIDATES:
            if info["finite"][name] and info["err"][name] * info["norm_kk"] <= self.b:
                return name
        return "fp64"


class GlobalPaced:
    """Method B-paced: the lowest format whose numerical contribution is within the budget left divided by the blocks left."""
    name = "B-paced: global budget"

    def __init__(self, m, nB, n_blocks, **kw):
        self.budget, self.used, self.left = (TAU * m)**2, 0.0, n_blocks

    def choose(self, info):
        allowance = (self.budget - self.used) / self.left
        self.left -= 1
        for name in CANDIDATES:
            if info["finite"][name] and info["d2"][name] <= allowance:
                self.used += info["d2"][name]
                return name
        return "fp64"


class GlobalPlanned:
    """Method B-optimal, second pass: the planned format when the running numerical budget allows it, else the next one."""
    name = "B-optimal: global budget"

    def __init__(self, m, nB, n_blocks, plan=None, **kw):
        self.budget, self.used, self.plan, self.fallbacks = (TAU * m)**2, 0.0, plan, 0

    def choose(self, info):
        start = FORMAT_NAMES.index(self.plan[info["i"], info["k"]])
        for name in FORMAT_NAMES[start:]:
            if info["finite"][name] and self.used + info["d2"][name] <= self.budget:
                if name != self.plan[info["i"], info["k"]]:
                    self.fallbacks += 1
                self.used += info["d2"][name]
                return name
        return "fp64"


class InferenceDriven:
    """Method C, Algorithm 2: two paced budgets, the numerical budget tau^2 m^2 on d^2 and the statistical budget epsilon on
    the inference-sensitivity score s; the lowest format within both allowances is stored, fp64 if none is."""

    def __init__(self, m, nB, n_blocks, epsilon=np.inf, **kw):
        self.budget_N, self.used_N = (TAU * m)**2, 0.0
        self.budget_S, self.used_S = epsilon, 0.0
        self.left, self.epsilon = n_blocks, epsilon
        self.name = f"C: inference-driven, eps={epsilon:g}"

    def choose(self, info):
        b_N = (self.budget_N - self.used_N) / self.left
        b_S = (self.budget_S - self.used_S) / self.left if np.isfinite(self.budget_S) else np.inf
        self.left -= 1
        for name in CANDIDATES:
            if info["finite"][name] and info["d2"][name] <= b_N and info["s"][name] <= b_S:
                self.used_N += info["d2"][name]
                self.used_S += info["s"][name]
                return name
        return "fp64"


class AllFp64:
    name = "D: fp64 reference"

    def __init__(self, *a, **kw):
        pass

    def choose(self, info):
        return "fp64"


def plan_optimal(costs, sizes, budget):
    """Offline allocation from recorded costs: every block starts in fp32, then blocks move to bf16 in order of increasing extra
    cost per byte saved until the budget is spent; the minimum-storage assignment for the recorded costs when the block sizes
    are equal."""
    plan, used = {}, 0.0
    for key in costs:
        plan[key] = "fp32" if np.isfinite(costs[key]["fp32"]) else "fp64"
        used += costs[key][plan[key]]
    if used > budget:
        for key in sorted(costs, key=lambda key: -costs[key]["fp32"]):
            if used <= budget:
                break
            if plan[key] == "fp32":
                used -= costs[key]["fp32"]; plan[key] = "fp64"
    order = sorted((key for key in costs if plan[key] == "fp32" and np.isfinite(costs[key]["bf16"])),
                   key=lambda key: (costs[key]["bf16"] - costs[key]["fp32"]) / (sizes[key] * (BYTES["fp32"] - BYTES["bf16"])))
    for key in order:
        extra = costs[key]["bf16"] - costs[key]["fp32"]
        if used + extra <= budget:
            used += extra; plan[key] = "bf16"
    return plan, used


# ----------------------------------------------------------------------------- the factorization: Algorithm 1 with the observed-data recurrences of Algorithm 2

def mixed_precision_cholesky(model, rule, record=None):
    """Block Cholesky factorization with the storage format of every off-diagonal block chosen by `rule`.

    The covariance blocks are formed in fp64 and discarded, all arithmetic is fp64, the diagonal factor blocks are kept in fp64.
    Alongside the recurrence it carries the block forward substitution of the observed residuals, z_k = L_kk^{-1} r_k and
    r_i <- r_i - L~_ik z_k, and the diagonal v_i of the Schur complement of every row block, so that the inference-sensitivity
    score of every candidate format is available to the rule before the block is stored. For every block the conversion
    errors, the numerical contributions d^2 and the scores s of all candidate formats are measured and passed to the rule; the
    contributions of the chosen format are accumulated. The time spent on measuring the candidates and choosing is reported as
    the selection overhead. If `record` is a dict, the costs of every block in every format are stored in it.
    """
    blocks, nB = model.blocks, len(model.blocks)
    y = model.y
    L, logdet, used_N, used_S, select_time = {}, 0.0, 0.0, 0.0, 0.0
    count = {name: 0 for name in FORMAT_NAMES}
    precision = np.full((nB, nB), np.nan)
    r = [y[b].clone() for b in blocks]                                             # observed-data residuals, per block
    v = [None] * nB                                                                # Schur-complement diagonals, per block
    z = [None] * nB                                                                # innovations z_k
    assignments, failure = [], None
    n_off = nB * (nB - 1) // 2
    left = n_off
    t_start = time.time()
    for k in range(nB):
        S = model.block(blocks[k], blocks[k])
        S.diagonal().add_(model.nugget[blocks[k]])
        if v[k] is None:
            v[k] = S.diagonal().clone()
        row = [L[k, j].to(torch.float64) for j in range(k)]
        for Lkj in row:
            S.addmm_(Lkj, Lkj.T, alpha=-1.0)
        try:
            Lkk = torch.linalg.cholesky(S)
        except Exception as error:                                                 # not expected under the guarantee; reported if it happens
            failure = f"diagonal block {k}: {str(error).splitlines()[0]}"
            break
        L[k, k] = Lkk
        precision[k, k] = FORMAT_NAMES.index("fp64")
        logdet += 2.0 * float(torch.log(torch.diagonal(Lkk)).sum())
        norm_kk = float(torch.linalg.matrix_norm(Lkk, ord=2))
        z[k] = torch.linalg.solve_triangular(Lkk, r[k].reshape(-1, 1), upper=False).reshape(-1)
        for i in range(k + 1, nB):
            if v[i] is None:
                v[i] = model.block(blocks[i], blocks[i]).diagonal() + model.nugget[blocks[i]]
            A = model.block(blocks[i], blocks[k])
            for j in range(k):
                A.addmm_(L[i, j].to(torch.float64), row[j].T, alpha=-1.0)
            A = torch.linalg.solve_triangular(Lkk.T, A, upper=True, left=False)
            t0 = time.time()
            E, err, d2, finite = block_costs(A, Lkk)
            r0 = r[i] - A @ z[k]                                                   # the fp64 candidate residual
            s, e_norm = {}, {}
            for name in FORMAT_NAMES:
                s[name], e_norm[name] = sensitivity(E[name], z[k], r0, v[i], model.m)
            info = dict(i=i, k=k, err=err, d2=d2, s=s, finite=finite, norm_kk=norm_kk, left=left)
            if record is not None:
                record[i, k] = dict(d2)
            name = rule.choose(info)
            select_time += time.time() - t0
            left -= 1
            Lik = A.to(CANDIDATES.get(name, torch.float64))
            L[i, k] = Lik
            Lik64 = Lik.to(torch.float64)
            r[i] = r[i] - Lik64 @ z[k]                                             # residual update with the stored block
            v[i] = v[i] - (Lik64 * Lik64).sum(dim=1)                               # Schur-complement diagonal proxy
            used_N += d2[name]
            used_S += s[name]
            count[name] += 1
            precision[i, k] = FORMAT_NAMES.index(name)
            assignments.append({"i": i, "k": k, "format": name, "d2": d2[name], "s": s[name], "e_norm": e_norm[name],
                                "d2_bf16": d2["bf16"], "d2_fp32": d2["fp32"], "s_bf16": s["bf16"], "s_fp32": s["fp32"],
                                "per_block_test_bf16": err["bf16"] * norm_kk, "per_block_test_fp32": err["fp32"] * norm_kk})
    elapsed = time.time() - t_start
    quadratic = float(sum((zk * zk).sum() for zk in z if zk is not None)) if failure is None else np.nan   # y^T Sigma~^{-1} y from the carried forward substitution
    return dict(L=L, logdet=logdet, count=count, precision=precision, used_N=used_N, used_S=used_S, bytes=sum(b.nbytes for b in L.values()),
                time=elapsed, select_time=select_time, assignments=assignments, failure=failure, quadratic=quadratic)


def solve(L, blocks, rhs):
    # solve (L L^T) x = rhs by block forward and back substitution; rhs is a vector or a matrix with one right-hand side per column
    x, nB = rhs.clone().reshape(rhs.shape[0], -1), len(blocks)
    for k in range(nB):
        s = x[blocks[k]]
        for j in range(k):
            s -= L[k, j].to(torch.float64) @ x[blocks[j]]
        x[blocks[k]] = torch.linalg.solve_triangular(L[k, k], s, upper=False)
    for k in range(nB - 1, -1, -1):
        s = x[blocks[k]]
        for i in range(k + 1, nB):
            s -= L[i, k].to(torch.float64).T @ x[blocks[i]]
        x[blocks[k]] = torch.linalg.solve_triangular(L[k, k].T, s, upper=True)
    return x.reshape(rhs.shape)


def negative_log_likelihood(theta, model):
    # Gaussian log likelihood of the replicates, which share the covariance matrix; one factorization, one solve per replicate
    model.set_parameters(theta)
    L, logdet, count, precision = mixed_precision_cholesky(model)
    alpha = solve(L, model.blocks, model.y)                                        # (N, replicates)
    N, T = model.N, model.T
    sigma2 = float((model.y * alpha).sum()) / (N * T)                              # profiled variance of the first output
    stored_gb = sum(block.nbytes for block in L.values()) / 1e9
    return 0.5 * (N * T * np.log(sigma2) + T * logdet + N * T + N * T * np.log(2 * np.pi)), sigma2, count, stored_gb, precision

def negative_log_likelihood_from_factor(model, result):
    # the profiled Gaussian log likelihood of the stacked residuals, evaluated with a given factor
    alpha = solve(result["L"], model.blocks, model.y)
    N = model.N
    sigma2 = float(model.y @ alpha) / N
    return 0.5 * (N * np.log(sigma2) + result["logdet"] + N + N * np.log(2 * np.pi)), sigma2

def predict(model, L, F_new, sigma2, chunk=8):
    # predictive mean and standard deviation of the three standardized component curves of every new stride, in chunks of strides
    alpha = solve(L, model.blocks, model.y)
    all_rows = torch.arange(model.N, device=DEVICE)
    means, sds = [], []
    eta = torch.tensor(model.parameters["eta"], device=DEVICE)
    for a in range(0, F_new.shape[0], chunk):
        K_star = model.cross(F_new[a:a + chunk], all_rows)                                                 # (3 n_chunk 101, N)
        n_chunk = min(chunk, F_new.shape[0] - a)
        mean = (K_star @ alpha).reshape(3, n_chunk, model.T)
        V = solve(L, model.blocks, K_star.T)                                                                # (N, 3 n_chunk 101)
        explained = (K_star * V.T).sum(dim=1).reshape(3, n_chunk, model.T)
        prior = (torch.diagonal(model.amplitude) + eta)[:, None, None]
        var = sigma2 * (prior - explained)
        means.append(mean.numpy()); sds.append(np.sqrt(np.maximum(var.numpy(), 1e-12)))
    return np.concatenate(means, axis=1), np.concatenate(sds, axis=1)

def dense_check(model, result):
    # forms L~ L~^T - Sigma, for verification only, and returns its Frobenius norm and the smallest eigenvalues
    blocks, nB = model.blocks, len(model.blocks)
    Sigma = torch.cat([torch.cat([model.block(blocks[i], blocks[j]) for j in range(nB)], dim=1) for i in range(nB)], dim=0)
    Sigma.diagonal().add_(model.nugget)
    Lt = torch.zeros_like(Sigma)
    for (i, k), block in result["L"].items():
        Lt[blocks[i], blocks[k]] = block.to(torch.float64)
    Delta = Lt @ Lt.T - Sigma
    return float(torch.linalg.norm(Delta)), float(torch.linalg.eigvalsh(Sigma)[0]), float(torch.linalg.eigvalsh(Lt @ Lt.T)[0])

# ----------------------------------------------------------------------------- model at fixed parameters

model = SpaceTimeTrivariateMatern(F_train, Z_train, GRID.astype(float), args.block_size)
N, nB = model.N, len(model.blocks)
n_off = nB * (nB - 1) // 2
sizes = {(i, k): (model.blocks[i].stop - model.blocks[i].start) * (model.blocks[k].stop - model.blocks[k].start) for k in range(nB) for i in range(k + 1, nB)}
print(f"covariance matrix          {N:,} x {N:,}   (1 joint x {len(COMPONENTS)} components x {model.n} strides x {model.T} gait points)")
print(f"matrix in fp64             {N * N * 8 / 1e9:.2f} GB, Cholesky factor in fp64 {N * (N + 1) / 2 * 8 / 1e9:.2f} GB")
print(f"blocks                     {nB} x {nB} of up to {args.block_size} points, {n_off} off-diagonal factor blocks")

l0 = np.full(3, {"hip": 0.354, "knee": 0.287, "ankle": 0.313}[args.joint])
lt0 = np.full(3, {"hip": 5.57, "knee": 4.34, "ankle": 4.46}[args.joint])
theta0 = np.concatenate([np.log(l0), np.log(lt0), [0.3, 0.3, 0.3], [0.0, 0.0], np.log([1e-3, 1e-3, 1e-3])])
bounds = [(np.log(0.05), np.log(5.0))] * 3 + [(np.log(1.0), np.log(30.0))] * 3 + [(-2.0, 2.0)] * 3 + [(-3.0, 3.0)] * 2 + [(np.log(1e-5), np.log(0.1))] * 3


def make_rule(name, m, epsilon=None, plan=None):
    if name == "A: per-block budget":
        return PerBlockRule(m, nB, n_off)
    if name == "B-paced: global budget":
        return GlobalPaced(m, nB, n_off)
    if name == "B-optimal: global budget":
        return GlobalPlanned(m, nB, n_off, plan=plan)
    if name == "D: fp64 reference":
        return AllFp64()
    return InferenceDriven(m, nB, n_off, epsilon=epsilon)


def fit_parameters(rule_name, epsilon, checkpoint, verbose):
    # maximum likelihood estimation, unchanged from the earlier scripts, with the named rule inside every likelihood evaluation
    if os.path.exists(checkpoint):
        saved = json.load(open(checkpoint))
        print(f"{rule_name}: parameters from {checkpoint}")
        return np.array(saved["theta"]), saved["nll"], saved.get("evaluations", np.nan), saved.get("time_s", np.nan)
    best = {"nll": np.inf, "theta": None, "evaluations": 0}
    t_fit = time.time()

    def objective(th):
        if best["theta"] is not None and np.max(np.abs(th - best["theta"])) > 0.7:
            return best["nll"] + 1e6 * float(np.max(np.abs(th - best["theta"])))
        t0 = time.time()
        model.set_parameters(th)
        r = mixed_precision_cholesky(model, make_rule(rule_name, model.m, epsilon))
        if r["failure"] is not None:
            return 1e30
        nll, sigma2 = negative_log_likelihood_from_factor(model, r)
        best["evaluations"] += 1
        if nll < best["nll"]:
            best.update(nll=nll, theta=th.copy())
            json.dump({"theta": th.tolist(), "nll": nll, "evaluations": best["evaluations"], "time_s": time.time() - t_fit}, open(checkpoint, "w"))
        if verbose or best["evaluations"] % 50 == 0:
            p = model.parameters
            print(f"{rule_name:34s} evaluation {best['evaluations']:4d}  negative log likelihood {nll:13.2f}  blocks {'/'.join(FORMAT_NAMES)} {'/'.join(str(c) for c in r['count'].values())}  [{time.time() - t0:.1f} s]  "
                  f"input ranges {np.round(p['l'], 3).tolist()}  temporal ranges {np.round(p['lt'], 2).tolist()}  cross-correlations fx-fy {p['rho'][0, 1]:.3f}, fx-fz {p['rho'][0, 2]:.3f}, fy-fz {p['rho'][1, 2]:.3f}", flush=True)
        return nll

    result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.max_iterations, "eps": 1e-3})
    elapsed = time.time() - t_fit
    json.dump({"theta": best["theta"].tolist(), "nll": best["nll"], "evaluations": best["evaluations"], "time_s": elapsed}, open(checkpoint, "w"))
    print(f"{rule_name}: {result.message}, {result.nit} iterations, {best['evaluations']} likelihood evaluations, {elapsed / 60:.1f} min; checkpoint {checkpoint}", flush=True)
    return best["theta"], best["nll"], best["evaluations"], elapsed


theta_path = args.theta or os.path.join("results", "global_budget_spacetime", f"spacetime_checkpoint_{args.joint}_xyz_{model.n}strides.json")
if os.path.exists(theta_path):
    theta = np.array(json.load(open(theta_path))["theta"])
    print(f"fixed parameters from {theta_path}")
else:
    print(f"no checkpoint at {theta_path}; fitting the parameters with Method A first")
    theta_path = os.path.join(args.out, f"spacetime_checkpoint_{args.joint}_xyz_{model.n}strides_A.json")
    theta, _, _, _ = fit_parameters("A: per-block budget", None, theta_path, verbose=True)
model.set_parameters(theta)
p = model.parameters
budget_N = (TAU * model.m)**2
print(f"input ranges               {np.round(p['l'], 3).tolist()}   temporal ranges {np.round(p['lt'], 2).tolist()}   relative nugget effects {[f'{x:.2e}' for x in p['eta']]}   m = {model.m:.3e}")
print(f"cross-correlations         fx-fy {p['rho'][0, 1]:.3f}, fx-fz {p['rho'][0, 2]:.3f}, fy-fz {p['rho'][1, 2]:.3f}")
print(f"per-block budget           b = tau m / n_B = {TAU * model.m / nB:.3e};   numerical budget tau^2 m^2 = {budget_N:.3e} (tau m = {TAU * model.m:.3e});   statistical tolerances epsilon = {EPSILONS}")

# ----------------------------------------------------------------------------- comparison at fixed parameters, one factor in memory at a time

import gc

all_block_entries = sum(sizes.values()) + sum((model.blocks[k].stop - model.blocks[k].start)**2 for k in range(nB))
fp64_bytes = all_block_entries * 8                                                  # the same blocks, all stored in fp64 (diagonal blocks as full squares)
results, rows, all_assignments, reference = {}, [], [], {}
run_dense = not args.no_dense and N <= 12000


def size(n_bytes):
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.2f} MB"


def evaluate(name, rule, epsilon=np.nan, record=None):
    r = mixed_precision_cholesky(model, rule, record)
    row = {"method": name, "epsilon": epsilon, "completed": r["failure"] is None, "failure": r["failure"] or "",
           **{f"blocks_{f}": r["count"][f] for f in FORMAT_NAMES}, **{f"share_{f}_%": 100 * r["count"][f] / n_off for f in FORMAT_NAMES},
           "stored_factor_bytes": r["bytes"], "stored_factor_MB": r["bytes"] / 1e6, "reduction_vs_fp64_%": 100 * (1 - r["bytes"] / fp64_bytes),
           "numerical_error_sqrt_sum_d2": np.sqrt(r["used_N"]), "numerical_error_over_tau_m": np.sqrt(r["used_N"]) / (TAU * model.m),
           "numerical_budget_used_%": 100 * r["used_N"] / budget_N, "within_numerical_budget": r["used_N"] <= budget_N,
           "statistical_sensitivity_sum_s": r["used_S"], "statistical_budget_used_%": 100 * r["used_S"] / epsilon if np.isfinite(epsilon) and epsilon > 0 else np.nan,
           "factorization_time_s": r["time"], "selection_overhead_s": r["select_time"], "quadratic_form_carried": r["quadratic"]}
    if isinstance(rule, GlobalPlanned):
        row["fallbacks_in_pass2"] = rule.fallbacks
    if r["failure"] is None:
        nll, sigma2 = negative_log_likelihood_from_factor(model, r)
        row["negative_log_likelihood"], row["profiled_sigma2"] = nll, sigma2
        row["quadratic_form_solve"] = sigma2 * N
        if args.test_strides > 0:
            mean, sd = predict(model, r["L"], F_test, sigma2)
            if "pred_mean" not in reference:
                reference["pred_mean"] = mean
            row["max_prediction_difference_vs_first_BW"] = float(np.max(np.abs(scale[:, None, None] * (mean - reference["pred_mean"]))))
        if run_dense:
            dn, lam_min, lam_min_tilde = dense_check(model, r)
            row.update(dense_frobenius_norm=dn, dense_over_tracked=dn / np.sqrt(r["used_N"]) if r["used_N"] > 0 else np.nan, lambda_min_Sigma=lam_min, lambda_min_Sigma_tilde=lam_min_tilde)
    results[name] = {k: r[k] for k in ("count", "precision", "used_N", "used_S", "bytes", "time", "failure")}
    all_assignments.append(pd.DataFrame(r["assignments"]).assign(method=name, epsilon=epsilon))
    del r
    gc.collect()
    print(f"{name:34s} blocks {'/'.join(str(row[f'blocks_{f}']) for f in FORMAT_NAMES)}  stored {size(row['stored_factor_bytes'])}  numerical {row['numerical_error_over_tau_m']:.4f} tau m  "
          f"sum s {row['statistical_sensitivity_sum_s']:.3e}  [{row['factorization_time_s']:.1f} s, of which selection {row['selection_overhead_s']:.1f} s]", flush=True)
    return row


costs = {}
if not args.no_reference:
    rows.append(evaluate("D: fp64 reference", AllFp64()))
rows.append(evaluate("A: per-block budget", PerBlockRule(model.m, nB, n_off), record=costs))
rows.append(evaluate("B-paced: global budget", GlobalPaced(model.m, nB, n_off)))
if not args.no_optimal:
    plan, planned_used = plan_optimal(costs, sizes, args.plan_fraction * budget_N)
    rule = GlobalPlanned(model.m, nB, n_off, plan=plan)
    row = evaluate("B-optimal: global budget", rule)
    row.update(planned_budget_used_pct=100 * planned_used / budget_N, planned_bf16=sum(1 for f in plan.values() if f == "bf16"))
    rows.append(row)
for eps in EPSILONS:
    rule = InferenceDriven(model.m, nB, n_off, epsilon=eps)
    rows.append(evaluate(rule.name, rule, epsilon=eps))

table = pd.DataFrame(rows).set_index("method")
if "D: fp64 reference" in table.index:
    table["nll_minus_fp64"] = table["negative_log_likelihood"] - table.loc["D: fp64 reference", "negative_log_likelihood"]
    table["abs_nll_error_vs_fp64"] = table["nll_minus_fp64"].abs()
table["reduction_vs_A_%"] = 100 * (1 - table["stored_factor_bytes"] / table.loc["A: per-block budget", "stored_factor_bytes"])
table.to_csv(os.path.join(args.out, "inference_driven_comparison.csv"))
pd.concat(all_assignments).to_csv(os.path.join(args.out, "inference_driven_blocks.csv"), index=False)

# ----------------------------------------------------------------------------- report

eps_main = args.epsilon_main
main_C = f"C: inference-driven, eps={eps_main:g}"
shown_cols = ["A: per-block budget", "B-paced: global budget"] + (["B-optimal: global budget"] if not args.no_optimal else []) + [main_C] + (["D: fp64 reference"] if not args.no_reference else [])
def fmt(name, key, f="{:.3f}"):
    v = table.loc[name, key] if key in table.columns else np.nan
    return "" if pd.isna(v) else f.format(v)
report = pd.DataFrame({name: {
    "bf16 blocks [%]": fmt(name, "share_bf16_%", "{:.1f}"), "fp32 blocks [%]": fmt(name, "share_fp32_%", "{:.1f}"), "fp64 blocks [%]": fmt(name, "share_fp64_%", "{:.1f}"),
    "total factor storage": size(table.loc[name, "stored_factor_bytes"]),
    "storage reduction vs fp64 [%]": fmt(name, "reduction_vs_fp64_%", "{:.1f}"), "storage reduction vs original [%]": fmt(name, "reduction_vs_A_%", "{:.1f}"),
    "negative log likelihood": fmt(name, "negative_log_likelihood", "{:,.3f}"), "likelihood difference vs fp64": fmt(name, "nll_minus_fp64", "{:+.4f}"),
    "numerical budget utilization [%]": fmt(name, "numerical_budget_used_%", "{:.2f}"), "numerical error / (tau m)": fmt(name, "numerical_error_over_tau_m", "{:.4f}"),
    "statistical sensitivity sum s": fmt(name, "statistical_sensitivity_sum_s", "{:.3e}"), "statistical budget utilization [%]": fmt(name, "statistical_budget_used_%", "{:.2f}"),
    "factorization time [s]": fmt(name, "factorization_time_s", "{:.2f}"), "of which selection overhead [s]": fmt(name, "selection_overhead_s", "{:.2f}"),
    "completed": "yes" if table.loc[name, "completed"] else "no"} for name in shown_cols})
report.to_csv(os.path.join(args.out, "inference_driven_report.csv"))
print(f"\nfp64 factor {size(fp64_bytes)} as stored ({nB} full diagonal blocks and {n_off} off-diagonal blocks); N = {N:,}; fixed parameters, same blocks, tau = {TAU}")
print(report.to_string())

sweep = table[table.index.str.startswith("C:")].copy()
sweep_cols = ["epsilon", "blocks_bf16", "blocks_fp32", "blocks_fp64", "share_bf16_%", "stored_factor_MB", "reduction_vs_A_%", "numerical_budget_used_%", "statistical_sensitivity_sum_s", "statistical_budget_used_%"] + (["nll_minus_fp64"] if "nll_minus_fp64" in table.columns else []) + ["factorization_time_s"]
sweep[sweep_cols].to_csv(os.path.join(args.out, "inference_driven_epsilon_sweep.csv"))
print("\nsensitivity to the statistical tolerance epsilon (Method C); eps = inf is the inactive statistical constraint and should reproduce B-paced")
print(sweep[sweep_cols].to_string(float_format=lambda v: f"{v:.4g}"))
if run_dense:
    print("\ndense verification (forms L~ L~^T - Sigma; for checking only)")
    print(table[["dense_frobenius_norm", "numerical_error_sqrt_sum_d2", "dense_over_tracked", "lambda_min_Sigma", "lambda_min_Sigma_tilde"]].to_string(float_format=lambda v: f"{v:.4e}"))
print(f"\ncarried quadratic form y^T Sigma~^-1 y against the solve with the stored factor, largest relative difference over the methods: "
      f"{float(((table['quadratic_form_carried'] - table['quadratic_form_solve']).abs() / table['quadratic_form_solve']).max()):.1e}")
a = all_assignments[1 if not args.no_reference else 0]
print(f"scores s of the off-diagonal blocks in the run of Method A: bf16 median {a['s_bf16'].median():.3e}, max {a['s_bf16'].max():.3e}, sum {a['s_bf16'].sum():.3e};  fp32 median {a['s_fp32'].median():.3e}, sum {a['s_fp32'].sum():.3e}")
print("The positive-definiteness guarantee assumes exact fp64 arithmetic and storage; the implementation uses finite-precision fp64. The score s is a sensitivity proxy, not a bound on the likelihood.")

# ----------------------------------------------------------------------------- optional: parameters estimated under each method

if args.fit_each:
    fit_rows = []
    for rule_name, eps in [("A: per-block budget", None), ("B-paced: global budget", None), (main_C, eps_main)]:
        tag = rule_name.split(":")[0].replace(" ", "") + (f"_eps{eps:g}" if eps is not None else "")
        th, nll, evaluations, elapsed = fit_parameters(rule_name, eps, os.path.join(args.out, f"spacetime_checkpoint_{args.joint}_xyz_{model.n}strides_{tag}.json"), verbose=False)
        model.set_parameters(th)
        r = mixed_precision_cholesky(model, make_rule(rule_name, model.m, eps))
        _, sigma2 = negative_log_likelihood_from_factor(model, r)
        pp = model.parameters
        fit_rows.append({"method": rule_name, "negative log likelihood": nll, "evaluations": evaluations, "fit time [min]": elapsed / 60,
                         **{f"input range {c}": pp["l"][j] for j, c in enumerate(COMPONENTS)}, **{f"temporal range {c}": pp["lt"][j] for j, c in enumerate(COMPONENTS)},
                         **{f"variance {c}": sigma2 * pp["v"][j] for j, c in enumerate(COMPONENTS)}, **{f"nugget effect {c}": sigma2 * pp["eta"][j] for j, c in enumerate(COMPONENTS)},
                         "cross-correlation fx-fy": pp["rho"][0, 1], "cross-correlation fx-fz": pp["rho"][0, 2], "cross-correlation fy-fz": pp["rho"][1, 2],
                         "blocks bf16 / fp32 / fp64 at the estimates": " / ".join(str(r["count"][f]) for f in FORMAT_NAMES), "stored factor": size(r["bytes"])})
        del r
    estimates = pd.DataFrame(fit_rows).set_index("method").T
    estimates.to_csv(os.path.join(args.out, "inference_driven_estimates_by_method.csv"))
    print("\nparameters estimated under each method")
    print(estimates.to_string())
    model.set_parameters(theta)

# ----------------------------------------------------------------------------- figures

PRECISION_COLORS = {"bf16": "#1FBEC0", "fp32": "#F26F22", "fp64": "#4D4D4D"}
METHOD_COLORS = {"A": "#4D4D4D", "B-paced": "#0072B2", "B-optimal": "#56B4E9", "C": "#D55E00", "D": "#999999"}
cmap = ListedColormap([PRECISION_COLORS[f] for f in FORMAT_NAMES]); cmap.set_bad("white")
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

# figure 1: blockwise precision maps
map_names = [n for n in ["A: per-block budget", "B-paced: global budget", "B-optimal: global budget", main_C] if n in results]
fig, axes = plt.subplots(1, len(map_names), figsize=(4.6 * len(map_names), 5.6), constrained_layout=True)
for ax, name in zip(np.atleast_1d(axes), map_names):
    rr = results[name]
    ax.imshow(np.ma.masked_invalid(rr["precision"]), cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    ax.set_title(f"{name}\n" + "   ".join(f"{f} {100 * rr['count'][f] / n_off:.0f}%" for f in FORMAT_NAMES) + f"\n{size(rr['bytes'])}, numerical error {np.sqrt(rr['used_N']) / (TAU * model.m):.2f} tau m", fontsize=10)
    ax.set_xlabel("block column $k$")
np.atleast_1d(axes)[0].set_ylabel("block row $i$")
fig.legend(handles=[Patch(facecolor=PRECISION_COLORS[f], label=f) for f in FORMAT_NAMES], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"Storage precision of the off-diagonal factor blocks: per-block, global and inference-driven rules\n($N$ = {N:,} = 3 components $\\times$ {model.n} strides $\\times$ {model.T} gait points, $n_b$ = {args.block_size}, {nB} $\\times$ {nB} blocks, $\\tau$ = {TAU}, diagonal blocks fp64)", fontsize=12, y=1.1)
fig.savefig(os.path.join(args.out, "fig1_precision_maps.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# figure 2: precision mix of every method
names = list(results)
labels = [n.replace(": ", "\n").replace("inference-driven, ", "") for n in names]
fig, ax = plt.subplots(figsize=(1.3 * len(names) + 3, 4.6), constrained_layout=True)
bottom = np.zeros(len(names))
for f in FORMAT_NAMES:
    share = np.array([100 * results[n]["count"][f] / n_off for n in names])
    ax.bar(range(len(names)), share, bottom=bottom, color=PRECISION_COLORS[f], label=f, width=0.65)
    for x, (sh, b0) in enumerate(zip(share, bottom)):
        if sh >= 5:
            ax.text(x, b0 + sh / 2, f"{sh:.0f}%", ha="center", va="center", fontsize=8, color="white")
    bottom += share
for x, n in enumerate(names):
    ax.text(x, 102, size(results[n]["bytes"]), ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("off-diagonal factor blocks [%]"); ax.set_ylim(0, 114); ax.set_yticks(range(0, 101, 20))
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
ax.set_title("Precision mix of the stored Cholesky factor, with the stored size above each bar", loc="left", pad=10)
fig.savefig(os.path.join(args.out, "fig2_precision_mix.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# figure 3: total factor storage
fig, ax = plt.subplots(figsize=(1.3 * len(names) + 3, 4.2), constrained_layout=True)
vals = np.array([results[n]["bytes"] for n in names]) / 1e6
cols = [METHOD_COLORS["D" if n.startswith("D") else "A" if n.startswith("A") else "B-paced" if n.startswith("B-paced") else "B-optimal" if n.startswith("B-opt") else "C"] for n in names]
ax.bar(range(len(names)), vals, color=cols, width=0.65)
for x, vv in enumerate(vals):
    ax.text(x, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=8)
ax.axhline(fp64_bytes / 1e6, color="black", lw=0.8, ls=":", label="all blocks in fp64")
ax.set_xticks(range(len(names))); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("stored Cholesky factor [MB]")
ax.legend(frameon=False, fontsize=8); ax.set_title("Total Cholesky factor storage, diagonal blocks included", loc="left")
fig.savefig(os.path.join(args.out, "fig3_storage.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# figure 4: likelihood accuracy against storage
if "abs_nll_error_vs_fp64" in table.columns:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for n in names:
        if n.startswith("D"):
            continue
        key = "A" if n.startswith("A") else "B-paced" if n.startswith("B-paced") else "B-optimal" if n.startswith("B-opt") else "C"
        x, yv = table.loc[n, "stored_factor_MB"], max(table.loc[n, "abs_nll_error_vs_fp64"], 1e-6)
        ax.scatter(x, yv, color=METHOD_COLORS[key], s=55 if key != "C" else 40, marker="o" if key != "C" else "s", zorder=3)
        ax.annotate(n.replace("inference-driven, ", "").replace(": ", "\n") if key != "C" else f"eps={table.loc[n, 'epsilon']:g}", (x, yv), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("stored Cholesky factor [MB]"); ax.set_ylabel("|negative log likelihood - fp64| [nats]")
    ax.grid(alpha=0.3, which="both"); ax.set_title("Likelihood approximation error against factor storage (squares: Method C over epsilon)", loc="left", fontsize=10)
    fig.savefig(os.path.join(args.out, "fig4_likelihood_vs_storage.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# figure 5: sensitivity to epsilon
finite_eps = [e for e in EPSILONS if np.isfinite(e)]
xs = list(range(len(EPSILONS)))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
bottom = np.zeros(len(EPSILONS))
for f in FORMAT_NAMES:
    share = np.array([100 * results[f"C: inference-driven, eps={e:g}"]["count"][f] / n_off for e in EPSILONS])
    axes[0].bar(xs, share, bottom=bottom, color=PRECISION_COLORS[f], label=f, width=0.65); bottom += share
axes[0].set_xticks(xs); axes[0].set_xticklabels([f"{e:g}" if np.isfinite(e) else "inf" for e in EPSILONS]); axes[0].set_xlabel("statistical tolerance $\\varepsilon$")
axes[0].set_ylabel("off-diagonal factor blocks [%]"); axes[0].set_ylim(0, 100); axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="lower left"); axes[0].set_title("Precision mix of Method C", loc="left")
st = [results[f"C: inference-driven, eps={e:g}"]["bytes"] / 1e6 for e in EPSILONS]
axes[1].plot(xs, st, "s-", color=METHOD_COLORS["C"], label="Method C")
for key, n in [("A", "A: per-block budget"), ("B-paced", "B-paced: global budget"), ("B-optimal", "B-optimal: global budget")]:
    if n in results:
        axes[1].axhline(results[n]["bytes"] / 1e6, color=METHOD_COLORS[key], ls="--", lw=1.2, label=n)
axes[1].set_xticks(xs); axes[1].set_xticklabels([f"{e:g}" if np.isfinite(e) else "inf" for e in EPSILONS]); axes[1].set_xlabel("statistical tolerance $\\varepsilon$")
axes[1].set_ylabel("stored Cholesky factor [MB]"); axes[1].grid(alpha=0.3); axes[1].legend(frameon=False, fontsize=8); axes[1].set_title("Storage of Method C against the numerical-only rules", loc="left")
if "abs_nll_error_vs_fp64" in table.columns:
    ax2 = axes[1].twinx()
    err = [max(table.loc[f"C: inference-driven, eps={e:g}", "abs_nll_error_vs_fp64"], 1e-6) for e in EPSILONS]
    ax2.plot(xs, err, "o:", color="#009E73", label="|nll - fp64| of Method C"); ax2.set_yscale("log"); ax2.set_ylabel("|negative log likelihood - fp64| [nats]", color="#009E73")
    ax2.legend(frameon=False, fontsize=8, loc="upper center")
fig.savefig(os.path.join(args.out, "fig5_epsilon_sensitivity.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"\ntables and figures saved to {args.out}")
