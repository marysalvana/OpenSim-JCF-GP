"""
Trivariate Gaussian process emulation of the three components of a joint contact force.

The anterior-posterior, axial and medio-lateral components of the knee joint
contact force are modeled jointly with a trivariate Matern model over the
stride inputs, fitted by maximum likelihood with the adaptive block
mixed-precision Cholesky factorization of Salvana (2026). The output of each
component is one scalar per stride, by default its value at the gait point
where the resultant joint contact force of the stride peaks, so the three
outputs of a stride form one force vector. All arithmetic runs in fp64 on the
CPU. The candidate storage formats of the factor blocks are bf16, fp32 and
fp64, lowest precision first.

Usage
  python3 codes/jcf-gp-knee-xyz-mac.py                          # knee, components at the resultant peak, N = 1,440
  python3 codes/jcf-gp-knee-xyz-mac.py --target extreme         # signed extreme of each component over the gait cycle
  python3 codes/jcf-gp-knee-xyz-mac.py --target 25              # components at gait point 25
  python3 codes/jcf-gp-knee-xyz-mac.py --joint hip              # the hip instead of the knee
Run from the project root, where data/x-train.csv and data/y-train.csv live.
"""

import argparse
import json
import os
import platform
import resource
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

# ----------------------------------------------------------------------------- arguments

parser = argparse.ArgumentParser(description="Trivariate Gaussian process emulator of the three components of a joint contact force over the stride inputs, fitted by maximum likelihood with a mixed-precision Cholesky factorization.")
parser.add_argument("--strides", type=int, default=480, help="number of training strides, evenly spaced over the 480 (default 480)")
parser.add_argument("--joint", default="knee", choices=["hip", "knee", "ankle"], help="joint whose force components are modeled (default knee)")
parser.add_argument("--target", default="at-peak", help="output per component and stride: at-peak (value at the gait point of the peak resultant), extreme (signed extreme over the gait cycle), mean, or a gait point 0 to 100 (default at-peak)")
parser.add_argument("--block-size", type=int, default=120, help="points per block of the factorization (default 120)")
parser.add_argument("--tau", type=float, default=0.99, help="error-budget parameter, 0 < tau < 1 (default 0.99)")
parser.add_argument("--max-iterations", type=int, default=60, help="L-BFGS-B iterations (default 60)")
parser.add_argument("--data", default="data", help="folder with x-train.csv and y-train.csv (default data)")
parser.add_argument("--out", default="results", help="folder for the checkpoint, the estimates and the figure (default results)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------------------- setup

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.float64)
MEMORY_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
print(f"{platform.machine()}, {MEMORY_GB:.0f} GB memory, {torch.get_num_threads()} CPU threads, torch {torch.__version__}")

# ----------------------------------------------------------------------------- data

x_train = pd.read_csv(os.path.join(args.data, "x-train.csv"))
y_train = pd.read_csv(os.path.join(args.data, "y-train.csv"))
n_all = len(x_train)
keep = np.unique(np.linspace(0, n_all - 1, min(args.strides, n_all)).round().astype(int))
x_train, y_train = x_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(drop=True)
print(f"training strides: {len(keep)} of {n_all}")

GRID = np.arange(101)
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
def target(df, component):
    C = curve(df, f"{args.joint}_jcf_{component}_BW")                             # (strides, 101 gait points), in BW
    if args.target == "at-peak":
        t_peak = curve(df, f"{args.joint}_jcf_BW").argmax(axis=1)                  # gait point of the peak resultant of each stride
        return C[np.arange(len(C)), t_peak]
    if args.target == "extreme":
        return C[np.arange(len(C)), np.abs(C).argmax(axis=1)]
    if args.target == "mean":
        return C.mean(axis=1)
    return C[:, int(args.target)]


Y_train = np.stack([target(y_train, component) for component in COMPONENTS])     # (3 components, strides), in BW
mean = Y_train.mean(axis=1)                                                       # mean of each component
scale = Y_train.std(axis=1)                                                       # s_j, in BW
Z_train = (Y_train - mean[:, None]) / scale[:, None]                              # standardized residuals
TARGET_LABEL = {"at-peak": f"{args.joint} JCF components at the peak resultant", "extreme": f"signed extreme of the {args.joint} JCF components",
                "mean": f"mean {args.joint} JCF components"}.get(args.target, f"{args.joint} JCF components at gait point {args.target}")

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
    def __init__(self, F, Z, block_size):
        self.n, self.N = Z.shape[1], Z.size
        Ft = torch.tensor(F, dtype=torch.float64, device=DEVICE)
        self.H = torch.cdist(Ft, Ft, compute_mode="donot_use_mm_for_euclid_dist") / F.shape[1]**0.5      # h between strides
        self.y = torch.tensor(Z.reshape(-1), dtype=torch.float64, device=DEVICE)                          # stacked residuals
        index = torch.arange(self.N, device=DEVICE)
        self.component, self.stride = index // self.n, index % self.n
        self.blocks = [slice(a, min(a + block_size, self.N)) for a in range(0, self.N, block_size)]

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


# ----------------------------------------------------------------------------- adaptive block mixed-precision Cholesky factorization
# Algorithm 1 of Salvana (2026). The precision rung is bf16, fp32, fp64. The diagonal
# blocks are kept in fp64, and each off-diagonal block is stored in the lowest format whose
# storage error fits the budget b = tau m / nB, where m = min_j eta_j bounds the smallest eigenvalue.

TAU = args.tau
CANDIDATES = {"bf16": torch.bfloat16, "fp32": torch.float32}      # lowest precision first
LARGEST = {"bf16": 3.39e38, "fp32": 3.40e38}                       # largest finite number of each format
FORMAT_NAMES = ["bf16", "fp32", "fp64"]


def choose_format(A, budget):
    # the lowest precision whose measured storage error fits the budget, fp64 if none does
    largest = float(A.abs().max())
    for name, dtype in CANDIDATES.items():
        if largest <= LARGEST[name] and float(torch.linalg.norm(A.to(dtype).to(torch.float64) - A)) <= budget:
            return name, dtype
    return "fp64", torch.float64


def mixed_precision_cholesky(model):
    blocks, nB = model.blocks, len(model.blocks)
    b = TAU * model.m / nB                                                         # budget of each block
    L, logdet = {}, 0.0
    count = {name: 0 for name in FORMAT_NAMES}
    precision = np.full((nB, nB), np.nan)                                          # format of every factor block, for the figure
    for k in range(nB):
        # diagonal block, kept in fp64
        S = model.block(blocks[k], blocks[k])
        S.diagonal().add_(model.nugget[blocks[k]])
        row = [L[k, j].to(torch.float64) for j in range(k)]
        for Lkj in row:
            S.addmm_(Lkj, Lkj.T, alpha=-1.0)
        Lkk = torch.linalg.cholesky(S)
        L[k, k] = Lkk
        precision[k, k] = FORMAT_NAMES.index("fp64")
        logdet += 2.0 * float(torch.log(torch.diagonal(Lkk)).sum())
        norm_kk = float(torch.linalg.matrix_norm(Lkk, ord=2))
        # off-diagonal blocks of column k, each stored in the lowest precision that fits the budget
        for i in range(k + 1, nB):
            A = model.block(blocks[i], blocks[k])
            for j in range(k):
                A.addmm_(L[i, j].to(torch.float64), row[j].T, alpha=-1.0)
            A = torch.linalg.solve_triangular(Lkk.T, A, upper=True, left=False)
            name, dtype = choose_format(A, b / norm_kk)
            L[i, k] = A.to(dtype)
            count[name] += 1
            precision[i, k] = FORMAT_NAMES.index(name)
    return L, logdet, count, precision


def solve(L, blocks, rhs):
    # solve (L L^T) x = rhs by block forward and back substitution
    x, nB = rhs.clone().reshape(-1, 1), len(blocks)
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
    model.set_parameters(theta)
    L, logdet, count, precision = mixed_precision_cholesky(model)
    alpha = solve(L, model.blocks, model.y)
    sigma2 = float(model.y @ alpha) / model.N                                      # profiled variance of the hip
    stored_gb = sum(block.nbytes for block in L.values()) / 1e9
    N = model.N
    return 0.5 * (N * np.log(sigma2) + logdet + N + N * np.log(2 * np.pi)), sigma2, count, stored_gb, precision


# ----------------------------------------------------------------------------- size of the covariance matrix

def size(n_bytes):
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.1f} MB"


model = TrivariateMatern(F_train, Z_train, args.block_size)
N, nB = model.N, len(model.blocks)
print(f"output                     {TARGET_LABEL} [BW]")
print(f"covariance matrix          {N:,} x {N:,}   (1 joint x {len(COMPONENTS)} components x {model.n} strides)")
print(f"number of entries          {N * N:,}")
print(f"matrix in fp64             {size(N * N * 8)}")
print(f"Cholesky factor in fp64    {size(N * (N + 1) / 2 * 8)}")
print(f"Cholesky factor in fp32    {size(N * (N + 1) / 2 * 4)}")
print(f"Cholesky factor in bf16    {size(N * (N + 1) / 2 * 2)}")
print(f"blocks                     {nB} x {nB}, each with up to {args.block_size:,} points")
print(f"precision rung             {' < '.join(FORMAT_NAMES)}, tau = {TAU}")
if N * (N + 1) / 2 * 8 / 1e9 > 0.5 * MEMORY_GB:
    print(f"warning: the fp64 Cholesky factor exceeds half of the {MEMORY_GB:.0f} GB of memory; reduce --strides")

# ----------------------------------------------------------------------------- maximum likelihood estimation

# starting values (fx, fy, fz): the univariate input range of the joint, unit variances of the standardized outputs, 1 % nugget effects
l0 = np.full(3, {"hip": 0.354, "knee": 0.287, "ankle": 0.313}[args.joint])   # input ranges [standardized units]
var0 = np.array([1.0, 1.0, 1.0])                           # variances, standardized
nugget0 = np.array([0.01, 0.01, 0.01])                     # nugget effects, standardized
theta0 = np.concatenate([np.log(l0), [0.3, 0.3, 0.3], np.log(var0[1:] / var0[0]), np.log(nugget0 / var0[0])])
bounds = [(np.log(0.05), np.log(5.0))] * 3 + [(-2.0, 2.0)] * 3 + [(-3.0, 3.0)] * 2 + [(np.log(1e-5), np.log(0.5))] * 3

tag = f"{args.joint}_xyz_{args.target}_{model.n}strides"
CHECKPOINT = os.path.join(args.out, f"trivariate_matern_components_checkpoint_{tag}.json")
if os.path.exists(CHECKPOINT):
    theta0 = np.array(json.load(open(CHECKPOINT))["theta"])
    print("resuming from", CHECKPOINT)

best = {"nll": np.inf, "theta": None, "evaluations": 0}
t_start = time.time()


def objective(theta):
    # a trial point far from the best point so far is rejected without a factorization
    if best["theta"] is not None and np.max(np.abs(theta - best["theta"])) > 0.7:
        return best["nll"] + 1e6 * float(np.max(np.abs(theta - best["theta"])))
    t0 = time.time()
    nll, sigma2, count, stored_gb, precision = negative_log_likelihood(theta, model)
    best["evaluations"] += 1
    if nll < best["nll"]:
        best.update(nll=nll, theta=theta.copy(), sigma2=sigma2, count=count, stored_gb=stored_gb, precision=precision)
        json.dump({"theta": theta.tolist(), "nll": nll}, open(CHECKPOINT, "w"))
    p = model.parameters
    print(f"evaluation {best['evaluations']:4d}  negative log likelihood {nll:13.2f}  blocks {'/'.join(FORMAT_NAMES)} {'/'.join(str(c) for c in count.values())}  [{time.time() - t0:.0f} s]\n"
          f"      input ranges {np.round(p['l'], 3).tolist()}  "
          f"variances [BW²] {np.round(sigma2 * p['v'] * scale**2, 4).tolist()}  nugget effects [BW²] {[f'{x:.2e}' for x in sigma2 * p['eta'] * scale**2]}  "
          f"cross-correlations fx-fy {p['rho'][0, 1]:.3f}, fx-fz {p['rho'][0, 2]:.3f}, fy-fz {p['rho'][1, 2]:.3f}", flush=True)
    return nll


result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.max_iterations, "eps": 1e-3})
print(f"{result.message}, {result.nit} iterations, {best['evaluations']} likelihood evaluations, {time.time() - t_start:.0f} s")

# ----------------------------------------------------------------------------- estimates

model.set_parameters(best["theta"])
p, sigma2 = model.parameters, best["sigma2"]
estimates = pd.DataFrame({"input range [standardized units]": p["l"],
                          "variance [BW²]": sigma2 * p["v"] * scale**2, "nugget effect [BW²]": sigma2 * p["eta"] * scale**2}, index=COMPONENTS).T
estimates.columns.name = f"{args.joint} JCF component"
shown = estimates.apply(lambda col: col.map("{:.3f}".format))
shown.loc["nugget effect [BW²]"] = estimates.loc["nugget effect [BW²]"].map("{:.2e}".format)
print(f"\nnegative log likelihood {best['nll']:,.1f}")
print(shown.to_string())
print("\ncross-correlations between the components at the same stride")
print(pd.DataFrame(p["rho"], index=COMPONENTS, columns=COMPONENTS).round(3).to_string())

peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9           # ru_maxrss is in bytes on macOS
summary = pd.Series({"N": f"{N:,}", "blocks": f"{nB} x {nB}",
                     f"off-diagonal blocks in {' / '.join(FORMAT_NAMES)}": " / ".join(str(c) for c in best["count"].values()),
                     "stored Cholesky factor": size(best['stored_gb'] * 1e9),
                     "Cholesky factor in fp64": size(N * (N + 1) / 2 * 8),
                     "covariance matrix in fp64": size(N * N * 8),
                     "peak process memory [GB]": f"{peak_gb:.2f}"}, name="mixed-precision factorization")
print("\n" + summary.to_frame().to_string())
estimates.to_csv(os.path.join(args.out, f"trivariate_matern_estimates_{tag}.csv"))
pd.DataFrame(p["rho"], index=COMPONENTS, columns=COMPONENTS).to_csv(os.path.join(args.out, f"trivariate_matern_cross_correlations_{tag}.csv"))

# ----------------------------------------------------------------------------- precision assigned to each block

PRECISION_COLORS = {"bf16": "#1FBEC0", "fp32": "#F26F22", "fp64": "#4D4D4D"}


def plot_precision(precision, count, N, block_size, path):
    nB, total = precision.shape[0], sum(count.values())
    cmap = ListedColormap([PRECISION_COLORS[name] for name in FORMAT_NAMES])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(6.4, 6.6), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(precision), cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N), origin="upper", interpolation="nearest")
    ax.set_title("   ".join(f"{name} {100 * count[name] / total:.1f}%" for name in FORMAT_NAMES), pad=12)
    ax.set_xlabel("block column $k$"); ax.set_ylabel("block row $i$")
    fig.legend(handles=[Patch(facecolor=PRECISION_COLORS[name], label=name) for name in FORMAT_NAMES], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Per-block precision of the factor\n(block size $n_b$ = {block_size}, {nB} $\\times$ {nB} blocks, $N$ = {N:,})", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


figure_path = os.path.join(args.out, f"trivariate_precision_per_block_{tag}.png")
plot_precision(best["precision"], best["count"], model.N, args.block_size, figure_path)
print(f"\nfigure saved to {figure_path}")
