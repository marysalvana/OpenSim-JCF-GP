"""
Global error-budget allocation for the adaptive block mixed-precision Cholesky factorization.

Benchmark on the trivariate Matern emulator of the three knee joint contact force
components (jcf-gp-knee-xyz-mac.py), at fixed, previously fitted parameters, of
two precision-assignment rules for the off-diagonal blocks of the Cholesky factor.

Method A, the original rule of Salvana (2026). Block (i, k) is stored in the
lowest format p whose conversion error satisfies
    ||fl_p(A_ik) - A_ik||_F ||L_kk||_2 <= b = tau m / n_B,
with fp64 as the fallback. Every block gets the same share of the budget.

Method B, the global rule. The perturbation of the reconstructed covariance
matrix is Delta = L~ L~^T - Sigma, whose diagonal blocks are zero and whose
off-diagonal blocks are Delta_ik = E_ik L_kk^T with E_ik = fl_p(A_ik) - A_ik.
Hence ||Delta||_F^2 = 2 sum_{i>k} ||E_ik L_kk^T||_F^2, and positive definiteness
is guaranteed by Weyl's inequality whenever
    2 sum_{i>k} ||E_ik L_kk^T||_F^2 <= tau^2 m^2.
A running total of the realized contributions d_ik,p^2 = 2 ||E_ik L_kk^T||_F^2
is kept during the factorization, and a candidate format is admissible only if
it keeps the total within tau^2 m^2. Three allocations of the global budget are
compared:
  B-greedy   first feasible format, blocks in factorization order
  B-paced    the lowest format whose contribution is within the budget left,
             divided equally over the blocks left (unused budget rolls over)
  B-optimal  two passes: a reference fp64 factorization records the cost of
             every format for every block; the blocks are then assigned bf16 in
             order of increasing cost until the budget is spent, which is the
             optimal assignment for equal block sizes; the second pass stores
             the planned formats and re-checks every contribution against the
             running total, falling back to the next format when the realized
             cost would exceed the budget.
In every variant a block is accepted only if the running total stays within
tau^2 m^2, so the induction of Propositions 1 and 2 of the paper applies
unchanged and the factorization completes, under the assumption of exact fp64
arithmetic and storage. The implementation runs in finite-precision fp64, whose
rounding is not covered by the bound.

Usage
  python3 codes/jcf-gp-knee-xyz-global-budget.py
  python3 codes/jcf-gp-knee-xyz-global-budget.py --block-size 60 --tau 0.99 --verify-dense
Run from the project root. Reads the fitted parameters from the checkpoint of
jcf-gp-knee-xyz-mac.py; writes CSV tables and PNG figures to --out.
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

# ----------------------------------------------------------------------------- arguments

parser = argparse.ArgumentParser(description="Compare the per-block and the global error budget for the mixed-precision Cholesky factorization of the trivariate knee JCF component model.")
parser.add_argument("--joint", default="knee", choices=["hip", "knee", "ankle"], help="joint whose force components are modeled (default knee)")
parser.add_argument("--strides", type=int, default=480, help="number of training strides, evenly spaced over the 480 (default 480)")
parser.add_argument("--step", type=int, default=1, help="use every step-th gait point as a replicate (default 1)")
parser.add_argument("--block-size", type=int, default=120, help="points per block of the factorization (default 120)")
parser.add_argument("--tau", type=float, default=0.99, help="error-budget parameter, 0 < tau < 1 (default 0.99)")
parser.add_argument("--theta", default=None, help="JSON checkpoint with the fitted parameters (default: the checkpoint of jcf-gp-knee-xyz-mac.py in --out's parent)")
parser.add_argument("--data", default="data", help="folder with the train and test files (default data)")
parser.add_argument("--out", default="results/global_budget", help="folder for the tables and figures (default results/global_budget)")
parser.add_argument("--verify-dense", action="store_true", help="also form the dense perturbation L~ L~^T - Sigma to confirm the tracked error and the smallest eigenvalue (small N only)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------------------- setup

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.float64)
print(f"{platform.machine()}, {torch.get_num_threads()} CPU threads, torch {torch.__version__}")

# ----------------------------------------------------------------------------- data

x_train = pd.read_csv(os.path.join(args.data, "x-train.csv"))
y_train = pd.read_csv(os.path.join(args.data, "y-train.csv"))
x_test = pd.read_csv(os.path.join(args.data, "x-test.csv"))
y_test = pd.read_csv(os.path.join(args.data, "y-test.csv"))
n_all = len(x_train)
keep = np.unique(np.linspace(0, n_all - 1, min(args.strides, n_all)).round().astype(int))
x_train, y_train = x_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(drop=True)

GRID = np.arange(101)
GAIT_POINTS = GRID[::args.step]
COMPONENTS = ["fx", "fy", "fz"]
COMPONENT_LABELS = {"fx": "anterior-posterior", "fy": "axial", "fz": "medio-lateral"}
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
scale_t = Y_train.std(axis=1)                                                          # s_j(t), in BW
Z_train = (Y_train - mean_t[:, None, :]) / scale_t[:, None, :]                         # standardized residuals

# ----------------------------------------------------------------------------- trivariate Matern model

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


class TrivariateMatern:
    def __init__(self, F, n_outputs, block_size):
        self.n, self.N = F.shape[0], n_outputs * F.shape[0]
        Ft = torch.tensor(F, dtype=torch.float64, device=DEVICE)
        self.H = torch.cdist(Ft, Ft, compute_mode="donot_use_mm_for_euclid_dist") / F.shape[1]**0.5      # h between strides
        index = torch.arange(self.N, device=DEVICE)
        self.component, self.stride = index // self.n, index % self.n
        self.blocks = [slice(a, min(a + block_size, self.N)) for a in range(0, self.N, block_size)]

    def set_outputs(self, Z):
        # Z has shape (outputs, strides, replicates); column t of self.y stacks the standardized residuals of replicate t
        self.T = Z.shape[2]
        self.y = torch.tensor(Z.reshape(-1, Z.shape[2]), dtype=torch.float64, device=DEVICE)             # (N, replicates)

    def set_parameters(self, theta):
        # theta = log input ranges (3), correlation angles (3), log variance ratios of fy and fz (2), log relative nugget effects (3)
        l = np.exp(theta[0:3])
        beta = correlation_matrix(theta[3:6])
        v = np.concatenate([[1.0], np.exp(theta[6:8])])
        eta = np.exp(theta[8:11])
        inverse_l, gamma = pair_ranges(l)
        rho = beta * gamma                                                         # cross-correlations between the components at the same stride
        as_tensor = lambda a: torch.tensor(a, dtype=torch.float64, device=DEVICE)
        self.amplitude = as_tensor(rho * np.sqrt(np.outer(v, v)))
        self.inverse_l = as_tensor(inverse_l)
        self.nugget = as_tensor(eta)[self.component]                               # relative nugget effect of every point
        self.m = float(eta.min())                                                  # lower bound on the smallest eigenvalue
        self.parameters = dict(l=l, v=v, eta=eta, rho=rho)

    def block(self, rows, cols):
        # block of the covariance matrix between two sets of points, without the nugget effect
        pair = (self.component[rows][:, None], self.component[cols][None, :])
        h = self.H[self.stride[rows]][:, self.stride[cols]]
        return self.amplitude[pair] * matern(h * self.inverse_l[pair])


# ----------------------------------------------------------------------------- storage formats and the two error budgets

TAU = args.tau
CANDIDATES = {"bf16": torch.bfloat16, "fp32": torch.float32}      # lowest precision first
LARGEST = {"bf16": 3.39e38, "fp32": 3.40e38}                       # largest finite number of each format
FORMAT_NAMES = ["bf16", "fp32", "fp64"]
BYTES = {"bf16": 2, "fp32": 4, "fp64": 8}


def block_costs(A, Lkk):
    """Conversion error of A in every candidate format.

    Returns, per format, the Frobenius norm of the conversion error ||fl_p(A) - A||_F, used by the
    per-block rule, and the realized contribution d^2 = 2 ||(fl_p(A) - A) L_kk^T||_F^2 to the squared
    Frobenius norm of the perturbation of the reconstructed covariance matrix, used by the global rule.
    A format whose conversion overflows is marked infeasible. fp64 is the reference and costs nothing.
    """
    largest = float(A.abs().max())
    err, d2, finite = {}, {}, {}
    for name, dtype in CANDIDATES.items():
        E = A.to(dtype).to(torch.float64) - A
        finite[name] = largest <= LARGEST[name] and bool(torch.isfinite(E).all())
        err[name] = float(torch.linalg.norm(E)) if finite[name] else np.inf
        d2[name] = 2.0 * float(torch.linalg.norm(E @ Lkk.T))**2 if finite[name] else np.inf
    err["fp64"], d2["fp64"], finite["fp64"] = 0.0, 0.0, True
    return err, d2, finite


class PerBlockRule:
    """Method A: the original rule, ||fl_p(A) - A||_F ||L_kk||_2 <= tau m / n_B."""
    name = "A: per-block budget"

    def __init__(self, m, nB, n_blocks):
        self.b = TAU * m / nB

    def choose(self, i, k, err, d2, finite, norm_kk):
        for name in CANDIDATES:
            if finite[name] and err[name] * norm_kk <= self.b:
                return name
        return "fp64"


class GlobalGreedy:
    """Method B-greedy: first feasible format under the running global budget, blocks in factorization order."""
    name = "B-greedy: global budget"

    def __init__(self, m, nB, n_blocks):
        self.budget, self.used = (TAU * m)**2, 0.0

    def choose(self, i, k, err, d2, finite, norm_kk):
        for name in CANDIDATES:
            if finite[name] and self.used + d2[name] <= self.budget:
                self.used += d2[name]
                return name
        return "fp64"


class GlobalPaced:
    """Method B-paced: the lowest format whose contribution is within the budget left divided by the blocks left."""
    name = "B-paced: global budget"

    def __init__(self, m, nB, n_blocks):
        self.budget, self.used, self.left = (TAU * m)**2, 0.0, n_blocks

    def choose(self, i, k, err, d2, finite, norm_kk):
        allowance = (self.budget - self.used) / self.left
        self.left -= 1
        for name in CANDIDATES:
            if finite[name] and d2[name] <= allowance:
                self.used += d2[name]
                return name
        return "fp64"


class GlobalPlanned:
    """Method B-optimal, second pass: store the planned format when the running global budget allows it, else the next one."""
    name = "B-optimal: global budget"

    def __init__(self, m, nB, n_blocks, plan):
        self.budget, self.used, self.plan, self.fallbacks = (TAU * m)**2, 0.0, plan, 0

    def choose(self, i, k, err, d2, finite, norm_kk):
        start = FORMAT_NAMES.index(self.plan[i, k])
        for name in FORMAT_NAMES[start:]:
            if finite[name] and self.used + d2[name] <= self.budget:
                if name != self.plan[i, k]:
                    self.fallbacks += 1
                self.used += d2[name]
                return name
        return "fp64"


def plan_optimal(costs, sizes, budget):
    """Offline allocation from the reference costs: every block starts in fp32, then blocks move to bf16 in order of
    increasing extra cost per byte saved until the budget is spent. With equal block sizes this is the minimum-storage
    assignment for the reference costs, since the bytes saved per block are then the same and only the costs differ."""
    plan, used = {}, 0.0
    for key in costs:
        plan[key] = "fp32" if np.isfinite(costs[key]["fp32"]) else "fp64"
        used += costs[key][plan[key]]
    if used > budget:                                                              # fp32 alone does not fit: demote the costliest fp32 blocks to fp64
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


# ----------------------------------------------------------------------------- the factorization with a pluggable precision rule

def mixed_precision_cholesky(model, rule, record=None):
    """Algorithm 1 of the paper with the storage format of every off-diagonal block chosen by `rule`.

    All arithmetic is fp64; the diagonal factor blocks are kept in fp64. For every off-diagonal block the conversion errors
    of all candidate formats are measured and passed to the rule, and the realized contribution d^2 of the chosen format is
    accumulated, so the Frobenius norm of L~ L~^T - Sigma is known exactly (under exact fp64 arithmetic) when the
    factorization ends. If `record` is a dict, the costs of every block in every format are stored in it.
    """
    blocks, nB = model.blocks, len(model.blocks)
    L, logdet, used = {}, 0.0, 0.0
    count = {name: 0 for name in FORMAT_NAMES}
    precision = np.full((nB, nB), np.nan)
    assignments, failure = [], None
    t0 = time.time()
    for k in range(nB):
        S = model.block(blocks[k], blocks[k])
        S.diagonal().add_(model.nugget[blocks[k]])
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
        for i in range(k + 1, nB):
            A = model.block(blocks[i], blocks[k])
            for j in range(k):
                A.addmm_(L[i, j].to(torch.float64), row[j].T, alpha=-1.0)
            A = torch.linalg.solve_triangular(Lkk.T, A, upper=True, left=False)
            err, d2, finite = block_costs(A, Lkk)
            if record is not None:
                record[i, k] = dict(d2)
            name = rule.choose(i, k, err, d2, finite, norm_kk)
            L[i, k] = A.to(CANDIDATES.get(name, torch.float64))
            used += d2[name]
            count[name] += 1
            precision[i, k] = FORMAT_NAMES.index(name)
            assignments.append({"i": i, "k": k, "format": name, "d2": d2[name], "d2_bf16": d2["bf16"], "d2_fp32": d2["fp32"],
                                "per_block_test_bf16": err["bf16"] * norm_kk, "per_block_test_fp32": err["fp32"] * norm_kk})
    elapsed = time.time() - t0
    stored_bytes = sum(block.nbytes for block in L.values())
    return dict(L=L, logdet=logdet, count=count, precision=precision, used=used, bytes=stored_bytes, time=elapsed,
                assignments=assignments, failure=failure)


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
    # the replicate likelihood evaluated with a given factor
    alpha = solve(result["L"], model.blocks, model.y)
    N, T = model.N, model.T
    sigma2 = float((model.y * alpha).sum()) / (N * T)
    return 0.5 * (N * T * np.log(sigma2) + T * result["logdet"] + N * T + N * T * np.log(2 * np.pi)), sigma2


def predict(model, L, F_new, sigma2):
    # predictive mean of the three standardized components of every new stride at every replicate, and their standard deviation,
    # which is the same at every replicate because the replicates share the covariance matrix
    Fn = torch.tensor(F_new, dtype=torch.float64, device=DEVICE)
    Ft = torch.tensor(F_train, dtype=torch.float64, device=DEVICE)
    h = torch.cdist(Fn, Ft, compute_mode="donot_use_mm_for_euclid_dist") / F_train.shape[1]**0.5    # (new strides, training strides)
    n_new = Fn.shape[0]
    rows = torch.arange(n_new * 3, device=DEVICE)
    new_component, new_stride = rows // n_new, rows % n_new
    pair = (new_component[:, None], model.component[None, :])
    K_star = model.amplitude[pair] * matern(h[new_stride][:, model.stride] * model.inverse_l[pair])     # (3 n_new, N)
    alpha = solve(L, model.blocks, model.y)                                                             # (N, replicates)
    mean = (K_star @ alpha).reshape(3, n_new, -1)                                                       # (3, new strides, replicates)
    V = solve(L, model.blocks, K_star.T)                                                                # Sigma^{-1} k_*, (N, 3 n_new)
    explained = (K_star * V.T).sum(dim=1).reshape(3, n_new)
    prior = torch.diagonal(model.amplitude)[:, None] + torch.tensor(model.parameters["eta"], device=DEVICE)[:, None]
    var = sigma2 * (prior - explained)
    return mean.numpy(), np.sqrt(np.maximum(var.numpy(), 1e-12))



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


# ----------------------------------------------------------------------------- model at the fitted parameters

model = TrivariateMatern(F_train, len(COMPONENTS), args.block_size)
model.set_outputs(Z_train[:, :, GAIT_POINTS])
theta_path = args.theta or os.path.join(os.path.dirname(args.out.rstrip("/")) or ".", f"trivariate_matern_components_checkpoint_{args.joint}_xyz_{model.n}strides_{len(GAIT_POINTS)}replicates.json")
if os.path.exists(theta_path):
    theta = np.array(json.load(open(theta_path))["theta"])
    print(f"fitted parameters from {theta_path}")
else:
    l0 = np.full(3, {"hip": 0.354, "knee": 0.287, "ankle": 0.313}[args.joint])
    theta = np.concatenate([np.log(l0), [0.3, 0.3, 0.3], [0.0, 0.0], np.log([0.01, 0.01, 0.01])])
    print(f"no checkpoint at {theta_path}; using the starting values of the fit")
model.set_parameters(theta)
N, nB = model.N, len(model.blocks)
n_off = nB * (nB - 1) // 2
sizes = {(i, k): (model.blocks[i].stop - model.blocks[i].start) * (model.blocks[k].stop - model.blocks[k].start) for k in range(nB) for i in range(k + 1, nB)}
p = model.parameters
print(f"covariance matrix          {N:,} x {N:,}   (1 joint x {len(COMPONENTS)} components x {model.n} strides), {len(GAIT_POINTS)} replicates")
print(f"blocks                     {nB} x {nB} of up to {args.block_size} points, {n_off} off-diagonal factor blocks")
print(f"input ranges               {np.round(p['l'], 3).tolist()}   relative nugget effects {[f'{x:.2e}' for x in p['eta']]}   m = {model.m:.3e}")
print(f"per-block budget           b = tau m / n_B = {TAU * model.m / nB:.3e}")
print(f"global budget              tau^2 m^2 = {(TAU * model.m)**2:.3e}   (tau m = {TAU * model.m:.3e})")

# ----------------------------------------------------------------------------- run the methods

results = {}
reference_costs = {}
t = time.time()
results["fp64 reference"] = mixed_precision_cholesky(model, type("AllFp64", (), {"name": "fp64", "choose": staticmethod(lambda i, k, err, d2, finite, norm_kk: "fp64")})(), record=reference_costs)
results["A: per-block budget"] = mixed_precision_cholesky(model, PerBlockRule(model.m, nB, n_off))
results["B-greedy: global budget"] = mixed_precision_cholesky(model, GlobalGreedy(model.m, nB, n_off))
results["B-paced: global budget"] = mixed_precision_cholesky(model, GlobalPaced(model.m, nB, n_off))
t_plan = time.time()
plan, planned_used = plan_optimal(reference_costs, sizes, (TAU * model.m)**2)
t_plan = time.time() - t_plan
planned_rule = GlobalPlanned(model.m, nB, n_off, plan)
results["B-optimal: global budget"] = mixed_precision_cholesky(model, planned_rule)
results["B-optimal: global budget"]["plan_count"] = {name: sum(1 for v in plan.values() if v == name) for name in FORMAT_NAMES}
results["B-optimal: global budget"]["plan_used"] = planned_used
results["B-optimal: global budget"]["fallbacks"] = planned_rule.fallbacks
results["B-optimal: global budget"]["time_pass1"] = results["fp64 reference"]["time"] + t_plan

# ----------------------------------------------------------------------------- report

fp64_bytes = results["fp64 reference"]["bytes"]                                  # the same blocks, all stored in fp64 (diagonal blocks as full squares)
ref_nll, ref_sigma2 = negative_log_likelihood_from_factor(model, results["fp64 reference"])
ref_mean, ref_sd = predict(model, results["fp64 reference"]["L"], F_test, ref_sigma2)
budget = (TAU * model.m)**2
rows = []
for name, r in results.items():
    nll, sigma2 = negative_log_likelihood_from_factor(model, r) if r["failure"] is None else (np.nan, np.nan)
    if r["failure"] is None:
        mean, sd = predict(model, r["L"], F_test, sigma2)
        pred_diff = float(np.max(np.abs(scale_t[:, None, GAIT_POINTS] * (mean - ref_mean))))
    else:
        pred_diff = np.nan
    row = {"method": name, "completed": r["failure"] is None, "failure": r["failure"] or "",
           **{f"blocks_{f}": r["count"][f] for f in FORMAT_NAMES}, **{f"share_{f}_%": 100 * r["count"][f] / n_off for f in FORMAT_NAMES},
           "stored_factor_MB": r["bytes"] / 1e6, "reduction_vs_fp64_%": 100 * (1 - r["bytes"] / fp64_bytes),
           "reduction_vs_A_%": 100 * (1 - r["bytes"] / results["A: per-block budget"]["bytes"]),
           "global_error_sqrt_sum_d2": np.sqrt(r["used"]), "error_over_tau_m": np.sqrt(r["used"]) / (TAU * model.m),
           "budget_used_%": 100 * r["used"] / budget, "within_budget": r["used"] <= budget,
           "factorization_time_s": r["time"], "negative_log_likelihood": nll, "nll_minus_fp64": nll - ref_nll,
           "max_prediction_difference_vs_fp64_BW": pred_diff}
    if "plan_count" in r:
        row.update({f"planned_{f}": r["plan_count"][f] for f in FORMAT_NAMES}, planned_budget_used_pct=100 * r["plan_used"] / budget,
                   fallbacks_in_pass2=r["fallbacks"], reference_pass_time_s=r["time_pass1"])
    if args.verify_dense and r["failure"] is None:
        dn, lam_min, lam_min_tilde = dense_check(model, r)
        row.update(dense_frobenius_norm=dn, dense_over_tracked=dn / np.sqrt(r["used"]) if r["used"] > 0 else np.nan, lambda_min_Sigma=lam_min, lambda_min_Sigma_tilde=lam_min_tilde)
    rows.append(row)
table = pd.DataFrame(rows).set_index("method")
table.to_csv(os.path.join(args.out, "mixed_precision_budget_comparison.csv"))
pd.concat([pd.DataFrame(r["assignments"]).assign(method=name) for name, r in results.items()]).to_csv(os.path.join(args.out, "mixed_precision_budget_blocks.csv"), index=False)

shown = pd.DataFrame({
    "blocks bf16 / fp32 / fp64": [" / ".join(str(r["count"][f]) for f in FORMAT_NAMES) for r in results.values()],
    "bf16 share [%]": [f"{100 * r['count']['bf16'] / n_off:.1f}" for r in results.values()],
    "stored factor [MB]": [f"{r['bytes'] / 1e6:.3f}" for r in results.values()],
    "reduction vs fp64 [%]": [f"{100 * (1 - r['bytes'] / fp64_bytes):.1f}" for r in results.values()],
    "reduction vs A [%]": [f"{100 * (1 - r['bytes'] / results['A: per-block budget']['bytes']):.1f}" for r in results.values()],
    "error / (tau m)": [f"{np.sqrt(r['used']) / (TAU * model.m):.4f}" for r in results.values()],
    "budget used [%]": [f"{100 * r['used'] / budget:.2f}" for r in results.values()],
    "time [s]": [f"{r['time']:.2f}" for r in results.values()],
    "completed": ["yes" if r["failure"] is None else "no" for r in results.values()],
    "nll - nll(fp64)": [f"{table.loc[name, 'nll_minus_fp64']:+.4f}" for name in results],
}, index=list(results))
print(f"\nfp64 factor {fp64_bytes / 1e6:.3f} MB as stored ({nB} full diagonal blocks and {n_off} off-diagonal blocks; {N * (N + 1) / 2 * 8 / 1e6:.3f} MB as a packed triangle); negative log likelihood with the fp64 factor {ref_nll:,.3f}")
print(shown.to_string())
r = results["B-optimal: global budget"]
print(f"\nB-optimal plan from the reference costs: bf16 / fp32 / fp64 = {' / '.join(str(r['plan_count'][f]) for f in FORMAT_NAMES)}, "
      f"planned budget use {100 * r['plan_used'] / budget:.2f} %, fallbacks in the second pass {r['fallbacks']}, reference pass {r['time_pass1']:.2f} s")
if args.verify_dense:
    print("\ndense verification (forms L~ L~^T - Sigma; for checking only)")
    print(table[["dense_frobenius_norm", "global_error_sqrt_sum_d2", "dense_over_tracked", "lambda_min_Sigma", "lambda_min_Sigma_tilde"]].to_string(float_format=lambda v: f"{v:.4e}"))
a = pd.DataFrame(results["fp64 reference"]["assignments"])
print(f"\nreference costs d^2 of the off-diagonal blocks: bf16 median {a['d2_bf16'].median():.3e}, min {a['d2_bf16'].min():.3e}, max {a['d2_bf16'].max():.3e};  "
      f"fp32 median {a['d2_fp32'].median():.3e}, max {a['d2_fp32'].max():.3e};  global budget {budget:.3e}")
print("The guarantee assumes exact fp64 arithmetic and storage; the implementation runs in finite-precision fp64, whose rounding is not covered by the bound.")

# ----------------------------------------------------------------------------- figures

PRECISION_COLORS = {"bf16": "#1FBEC0", "fp32": "#F26F22", "fp64": "#4D4D4D"}
cmap = ListedColormap([PRECISION_COLORS[f] for f in FORMAT_NAMES]); cmap.set_bad("white")
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
names = ["A: per-block budget", "B-greedy: global budget", "B-paced: global budget", "B-optimal: global budget"]
fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.6), constrained_layout=True)
for ax, name in zip(axes, names):
    r = results[name]
    ax.imshow(np.ma.masked_invalid(r["precision"]), cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    ax.set_title(f"{name}\n" + "   ".join(f"{f} {100 * r['count'][f] / n_off:.0f}%" for f in FORMAT_NAMES) + f"\n{r['bytes'] / 1e6:.2f} MB, error {np.sqrt(r['used']) / (TAU * model.m):.2f} tau m", fontsize=10)
    ax.set_xlabel("block column $k$")
axes[0].set_ylabel("block row $i$")
fig.legend(handles=[Patch(facecolor=PRECISION_COLORS[f], label=f) for f in FORMAT_NAMES], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"Storage precision of the off-diagonal factor blocks under the per-block and the global error budget\n($N$ = {N:,}, $n_b$ = {args.block_size}, {nB} $\\times$ {nB} blocks, $\\tau$ = {TAU}, diagonal blocks fp64)", fontsize=12, y=1.1)
fig.savefig(os.path.join(args.out, "mixed_precision_budget_precision_maps.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
bottom = np.zeros(len(names))
for f in FORMAT_NAMES:
    share = np.array([100 * results[n]["count"][f] / n_off for n in names])
    ax.bar(range(len(names)), share, bottom=bottom, color=PRECISION_COLORS[f], label=f, width=0.6)
    for x, (s, b0) in enumerate(zip(share, bottom)):
        if s >= 4:
            ax.text(x, b0 + s / 2, f"{s:.0f}%", ha="center", va="center", fontsize=9, color="white")
    bottom += share
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(": ", "\n") for n in names], fontsize=9)
ax.set_ylabel("off-diagonal factor blocks [%]"); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
for x, n in enumerate(names):
    ax.text(x, 102, f"{results[n]['bytes'] / 1e6:.2f} MB", ha="center", va="bottom", fontsize=9)
ax.set_title("Precision mix of the stored Cholesky factor, with the stored size above each bar", loc="left", pad=10)
fig.savefig(os.path.join(args.out, "mixed_precision_budget_precision_mix.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\ntables and figures saved to {args.out}")
