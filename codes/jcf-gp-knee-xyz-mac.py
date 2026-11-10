"""
Trivariate Gaussian process emulation of the three components of a joint contact force, gait point by gait point.

The anterior-posterior, axial and medio-lateral components of the knee joint
contact force are modeled jointly with a trivariate Matern model over the
stride inputs. One trivariate model is fitted at each of the 101 gait points,
by maximum likelihood with the adaptive block mixed-precision Cholesky
factorization of Salvana (2026), warm-started from the previous gait point.
The fitted models predict the three component curves of the held-out strides,
with 95 % prediction intervals. All arithmetic runs in fp64 on the CPU. The
candidate storage formats of the factor blocks are bf16, fp32 and fp64, lowest
precision first.

Usage
  python3 codes/jcf-gp-knee-xyz-mac.py                          # knee, all 101 gait points, N = 1,440 per fit
  python3 codes/jcf-gp-knee-xyz-mac.py --step 5                 # every fifth gait point
  python3 codes/jcf-gp-knee-xyz-mac.py --joint hip              # the hip instead of the knee
Run from the project root, where data/x-train.csv, y-train.csv, x-test.csv and y-test.csv live.
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
from scipy.stats import norm

# ----------------------------------------------------------------------------- arguments

parser = argparse.ArgumentParser(description="Trivariate Gaussian process emulator of the three components of a joint contact force, one fit per gait point, with a mixed-precision Cholesky factorization.")
parser.add_argument("--joint", default="knee", choices=["hip", "knee", "ankle"], help="joint whose force components are modeled (default knee)")
parser.add_argument("--strides", type=int, default=480, help="number of training strides, evenly spaced over the 480 (default 480)")
parser.add_argument("--step", type=int, default=1, help="fit every step-th gait point, 1 for all 101 (default 1)")
parser.add_argument("--block-size", type=int, default=120, help="points per block of the factorization (default 120)")
parser.add_argument("--tau", type=float, default=0.99, help="error-budget parameter, 0 < tau < 1 (default 0.99)")
parser.add_argument("--max-iterations", type=int, default=60, help="L-BFGS-B iterations per gait point (default 60)")
parser.add_argument("--data", default="data", help="folder with the train and test files (default data)")
parser.add_argument("--out", default="results", help="folder for the checkpoint, the estimates, the predictions and the figures (default results)")
parser.add_argument("--verbose", action="store_true", help="print every likelihood evaluation")
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
x_test = pd.read_csv(os.path.join(args.data, "x-test.csv"))
y_test = pd.read_csv(os.path.join(args.data, "y-test.csv"))
n_all = len(x_train)
keep = np.unique(np.linspace(0, n_all - 1, min(args.strides, n_all)).round().astype(int))
x_train, y_train = x_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(drop=True)
print(f"training strides: {len(keep)} of {n_all}; test strides: {len(x_test)}")

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
        self.y = torch.tensor(Z.reshape(-1), dtype=torch.float64, device=DEVICE)                          # stacked standardized residuals

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


model = TrivariateMatern(F_train, len(COMPONENTS), args.block_size)
N, nB = model.N, len(model.blocks)
print(f"output                     {args.joint} JCF components fx, fy, fz [BW], one trivariate fit at each of {len(GAIT_POINTS)} gait points")
print(f"covariance matrix          {N:,} x {N:,} per gait point   (1 joint x {len(COMPONENTS)} components x {model.n} strides)")
print(f"number of entries          {N * N:,}")
print(f"matrix in fp64             {size(N * N * 8)}")
print(f"Cholesky factor in fp64    {size(N * (N + 1) / 2 * 8)}")
print(f"Cholesky factor in fp32    {size(N * (N + 1) / 2 * 4)}")
print(f"Cholesky factor in bf16    {size(N * (N + 1) / 2 * 2)}")
print(f"blocks                     {nB} x {nB}, each with up to {args.block_size:,} points")
print(f"precision rung             {' < '.join(FORMAT_NAMES)}, tau = {TAU}")

# ----------------------------------------------------------------------------- maximum likelihood estimation at every gait point

# starting values at the first gait point (fx, fy, fz): the univariate input range of the joint, unit variances of the standardized outputs,
# 1 % nugget effects; every later gait point starts from the estimates of the previous one
l0 = np.full(3, {"hip": 0.354, "knee": 0.287, "ankle": 0.313}[args.joint])   # input ranges [standardized units]
var0 = np.array([1.0, 1.0, 1.0])                           # variances, standardized
nugget0 = np.array([0.01, 0.01, 0.01])                     # nugget effects, standardized
theta_first = np.concatenate([np.log(l0), [0.3, 0.3, 0.3], np.log(var0[1:] / var0[0]), np.log(nugget0 / var0[0])])
bounds = [(np.log(0.05), np.log(5.0))] * 3 + [(-2.0, 2.0)] * 3 + [(-3.0, 3.0)] * 2 + [(np.log(1e-5), np.log(0.5))] * 3

tag = f"{args.joint}_xyz_{model.n}strides"
CHECKPOINT = os.path.join(args.out, f"trivariate_matern_components_checkpoint_{tag}.json")
fits = {int(k): v for k, v in json.load(open(CHECKPOINT)).items()} if os.path.exists(CHECKPOINT) else {}
if fits:
    print(f"resuming from {CHECKPOINT}: {len(fits)} gait points done")


def fit_gait_point(t, theta0):
    # maximum likelihood estimation of the trivariate model for the outputs at gait point t
    model.set_outputs(Z_train[:, :, t])
    best = {"nll": np.inf, "theta": None, "evaluations": 0}

    def objective(theta):
        # a trial point far from the best point so far is rejected without a factorization
        if best["theta"] is not None and np.max(np.abs(theta - best["theta"])) > 0.7:
            return best["nll"] + 1e6 * float(np.max(np.abs(theta - best["theta"])))
        nll, sigma2, count, stored_gb, precision = negative_log_likelihood(theta, model)
        best["evaluations"] += 1
        if nll < best["nll"]:
            best.update(nll=nll, theta=theta.copy(), sigma2=sigma2, count=count, stored_gb=stored_gb, precision=precision)
        if args.verbose:
            p = model.parameters
            print(f"   t = {t:3d}  evaluation {best['evaluations']:4d}  negative log likelihood {nll:11.2f}  blocks {'/'.join(FORMAT_NAMES)} {'/'.join(str(c) for c in count.values())}  "
                  f"input ranges {np.round(p['l'], 3).tolist()}  cross-correlations fx-fy {p['rho'][0, 1]:.3f}, fx-fz {p['rho'][0, 2]:.3f}, fy-fz {p['rho'][1, 2]:.3f}", flush=True)
        return nll

    result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.max_iterations, "eps": 1e-3})
    best["iterations"], best["message"] = int(result.nit), str(result.message)
    return best


t_start = time.time()
theta_prev = theta_first
for t in GAIT_POINTS:
    if t in fits:
        theta_prev = np.array(fits[t]["theta"])
        continue
    t0 = time.time()
    best = fit_gait_point(t, theta_prev)
    model.set_parameters(best["theta"])
    p = model.parameters
    fits[t] = {"theta": best["theta"].tolist(), "nll": best["nll"], "sigma2": best["sigma2"], "count": best["count"],
               "stored_gb": best["stored_gb"], "iterations": best["iterations"], "evaluations": best["evaluations"], "message": best["message"]}
    json.dump({str(k): v for k, v in fits.items()}, open(CHECKPOINT, "w"))
    theta_prev = best["theta"]
    print(f"gait point {t:3d}  negative log likelihood {best['nll']:10.2f}  {best['iterations']:3d} iterations  {best['evaluations']:4d} evaluations  "
          f"blocks {'/'.join(FORMAT_NAMES)} {'/'.join(str(c) for c in best['count'].values())}  "
          f"input ranges {np.round(p['l'], 3).tolist()}  variances [BW²] {np.round(best['sigma2'] * p['v'] * scale_t[:, t]**2, 4).tolist()}  "
          f"nugget effects [BW²] {[f'{x:.1e}' for x in best['sigma2'] * p['eta'] * scale_t[:, t]**2]}  "
          f"cross-correlations fx-fy {p['rho'][0, 1]:.3f}, fx-fz {p['rho'][0, 2]:.3f}, fy-fz {p['rho'][1, 2]:.3f}  [{time.time() - t0:.0f} s]", flush=True)
print(f"{len(GAIT_POINTS)} gait points fitted, {time.time() - t_start:.0f} s")

# ----------------------------------------------------------------------------- estimates over the gait cycle

rows = []
for t in GAIT_POINTS:
    model.set_parameters(np.array(fits[t]["theta"]))
    p, sigma2 = model.parameters, fits[t]["sigma2"]
    row = {"gait_point": t, "negative_log_likelihood": fits[t]["nll"], "iterations": fits[t]["iterations"], "evaluations": fits[t]["evaluations"]}
    for j, c in enumerate(COMPONENTS):
        row[f"input_range_{c}"] = p["l"][j]
        row[f"variance_{c}_BW2"] = sigma2 * p["v"][j] * scale_t[j, t]**2
        row[f"nugget_{c}_BW2"] = sigma2 * p["eta"][j] * scale_t[j, t]**2
    row.update({"rho_fx_fy": p["rho"][0, 1], "rho_fx_fz": p["rho"][0, 2], "rho_fy_fz": p["rho"][1, 2]})
    row.update({f"blocks_{name}": fits[t]["count"][name] for name in FORMAT_NAMES})
    rows.append(row)
estimates = pd.DataFrame(rows).set_index("gait_point")
estimates.to_csv(os.path.join(args.out, f"trivariate_matern_estimates_by_gait_point_{tag}.csv"))

t_peak = int(np.median(curve(y_train, f"{args.joint}_jcf_BW").argmax(axis=1)))         # gait point of the peak resultant of the median stride
t_show = min(GAIT_POINTS, key=lambda t: abs(t - t_peak))
e = estimates.loc[t_show]
print(f"\nestimates at gait point {t_show}, the peak of the resultant   (negative log likelihood {e['negative_log_likelihood']:,.1f})")
shown = pd.DataFrame({c: {"input range [standardized units]": f"{e[f'input_range_{c}']:.3f}", "variance [BW²]": f"{e[f'variance_{c}_BW2']:.4f}",
                          "nugget effect [BW²]": f"{e[f'nugget_{c}_BW2']:.2e}"} for c in COMPONENTS})
shown.columns.name = f"{args.joint} JCF component"
print(shown.to_string())
print(f"cross-correlations at gait point {t_show}: fx-fy {e['rho_fx_fy']:.3f}, fx-fz {e['rho_fx_fz']:.3f}, fy-fz {e['rho_fy_fz']:.3f}")
print(f"\nover the {len(GAIT_POINTS)} gait points: input ranges {estimates[[f'input_range_{c}' for c in COMPONENTS]].median().round(3).tolist()} (median),  "
      f"cross-correlations fx-fy {estimates['rho_fx_fy'].min():.2f} to {estimates['rho_fx_fy'].max():.2f}, fx-fz {estimates['rho_fx_fz'].min():.2f} to {estimates['rho_fx_fz'].max():.2f}, "
      f"fy-fz {estimates['rho_fy_fz'].min():.2f} to {estimates['rho_fy_fz'].max():.2f}")
total_blocks = estimates[[f"blocks_{name}" for name in FORMAT_NAMES]].sum()
print(f"off-diagonal factor blocks over all fits, {' / '.join(FORMAT_NAMES)}: {' / '.join(str(int(v)) for v in total_blocks)}")

# ----------------------------------------------------------------------------- prediction of the test strides, gait point by gait point


def predict(model, L, F_new, sigma2):
    # predictive mean and standard deviation of the three standardized components of every new stride
    Fn = torch.tensor(F_new, dtype=torch.float64, device=DEVICE)
    Ft = torch.tensor(F_train, dtype=torch.float64, device=DEVICE)
    h = torch.cdist(Fn, Ft, compute_mode="donot_use_mm_for_euclid_dist") / F_train.shape[1]**0.5    # (new strides, training strides)
    n_new = Fn.shape[0]
    rows = torch.arange(n_new * 3, device=DEVICE)
    new_component, new_stride = rows // n_new, rows % n_new
    pair = (new_component[:, None], model.component[None, :])
    K_star = model.amplitude[pair] * matern(h[new_stride][:, model.stride] * model.inverse_l[pair])     # (3 n_new, N)
    alpha = solve(L, model.blocks, model.y).reshape(-1)
    mean = (K_star @ alpha).reshape(3, n_new)
    V = solve(L, model.blocks, K_star.T)                                                                # Sigma^{-1} k_*, (N, 3 n_new)
    explained = (K_star * V.T).sum(dim=1).reshape(3, n_new)
    prior = torch.diagonal(model.amplitude)[:, None] + torch.tensor(model.parameters["eta"], device=DEVICE)[:, None]
    var = sigma2 * (prior - explained)
    return mean.numpy(), np.sqrt(np.maximum(var.numpy(), 1e-12))


pred_mean = np.full(Y_test.shape, np.nan)                                               # (3 components, test strides, 101 gait points), in BW
pred_sd = np.full(Y_test.shape, np.nan)
precision_show = None
for t in GAIT_POINTS:
    model.set_outputs(Z_train[:, :, t])
    model.set_parameters(np.array(fits[t]["theta"]))
    L_fit, _, count, precision = mixed_precision_cholesky(model)
    z_mean, z_sd = predict(model, L_fit, F_test, fits[t]["sigma2"])
    pred_mean[:, :, t] = mean_t[:, t][:, None] + scale_t[:, t][:, None] * z_mean
    pred_sd[:, :, t] = scale_t[:, t][:, None] * z_sd
    if t == t_show:
        precision_show, count_show = precision, count
np.savez_compressed(os.path.join(args.out, f"trivariate_matern_predictions_{tag}.npz"), gait_points=GAIT_POINTS, pred_mean=pred_mean, pred_sd=pred_sd, y_test=Y_test,
                    subject=x_test["subject"].to_numpy(), block=x_test["block"].to_numpy(), stride_in_block=x_test["stride_in_block"].to_numpy())


def crps(y, mu, sd):
    z = (y - mu) / sd
    return np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


G = GAIT_POINTS
inside = np.abs(Y_test[:, :, G] - pred_mean[:, :, G]) <= 1.96 * pred_sd[:, :, G]
scores = pd.DataFrame({"RMSE [BW]": np.sqrt(np.mean((pred_mean[:, :, G] - Y_test[:, :, G])**2, axis=(1, 2))),
                       "CRPS [BW]": [crps(Y_test[j][:, G], pred_mean[j][:, G], pred_sd[j][:, G]) for j in range(3)],
                       "95% coverage [%]": inside.mean(axis=(1, 2)) * 100}, index=COMPONENTS).T
scores.columns.name = f"{args.joint} JCF component"
scores.to_csv(os.path.join(args.out, f"trivariate_matern_scores_{tag}.csv"))
print(f"\nout-of-sample scores on the {len(x_test)} test strides over {len(G)} gait points")
print(scores.round(3).to_string())

peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9           # ru_maxrss is in bytes on macOS
summary = pd.Series({"N per gait point": f"{N:,}", "gait points fitted": len(GAIT_POINTS), "blocks": f"{nB} x {nB}",
                     f"off-diagonal blocks in {' / '.join(FORMAT_NAMES)}, all fits": " / ".join(str(int(v)) for v in total_blocks),
                     "stored Cholesky factor per fit": size(np.mean([fits[t]["stored_gb"] for t in GAIT_POINTS]) * 1e9),
                     "Cholesky factor in fp64": size(N * (N + 1) / 2 * 8), "covariance matrix in fp64": size(N * N * 8),
                     "peak process memory [GB]": f"{peak_gb:.2f}"}, name="mixed-precision factorization")
print("\n" + summary.to_frame().to_string())

# ----------------------------------------------------------------------------- figures

PRECISION_COLORS = {"bf16": "#1FBEC0", "fp32": "#F26F22", "fp64": "#4D4D4D"}


def plot_precision(precision, count, N, block_size, path, t):
    nB, total = precision.shape[0], sum(count.values())
    cmap = ListedColormap([PRECISION_COLORS[name] for name in FORMAT_NAMES])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(6.4, 6.6), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(precision), cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N), origin="upper", interpolation="nearest")
    ax.set_title("   ".join(f"{name} {100 * count[name] / total:.1f}%" for name in FORMAT_NAMES), pad=12)
    ax.set_xlabel("block column $k$"); ax.set_ylabel("block row $i$")
    fig.legend(handles=[Patch(facecolor=PRECISION_COLORS[name], label=name) for name in FORMAT_NAMES], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Per-block precision of the factor at gait point {t}\n(block size $n_b$ = {block_size}, {nB} $\\times$ {nB} blocks, $N$ = {N:,})", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curves(path):
    # one stride per held-out participant, the one with the median error, for each component
    subjects_test = sorted(x_test["subject"].unique())
    labels = {"fx": "$f_x$ anterior-posterior", "fy": "$f_y$ axial", "fz": "$f_z$ medio-lateral"}
    fig, axes = plt.subplots(3, len(subjects_test), figsize=(6 * len(subjects_test), 11), sharex=True)
    for j, c in enumerate(COMPONENTS):
        y, mu, sd = Y_test[j][:, G], pred_mean[j][:, G], pred_sd[j][:, G]
        for col, s in enumerate(subjects_test):
            rows = np.where(x_test["subject"].to_numpy() == s)[0]
            err = np.sqrt(np.mean((mu[rows] - y[rows])**2, axis=1))
            i = rows[np.argsort(err)[len(err) // 2]]
            ax = axes[j][col]
            ax.fill_between(G, mu[i] - 1.96 * sd[i], mu[i] + 1.96 * sd[i], color="#f3c9b0", lw=0, label="95% interval")
            ax.plot(G, y[i], color="#1f3b5c", lw=2.2, label="OpenSim (Truth)")
            ax.plot(G, mu[i], color="#D55E00", lw=2.2, ls="--", label="emulator")
            ax.set_title(f"{c}: {s.replace('NIH_ACL_', '')}, {x_test.loc[i, 'block']}, stride {x_test.loc[i, 'stride_in_block']}, {x_test.loc[i, 'belt_speed_mps']:g} m/s", loc="left", fontsize=10)
            ax.text(0.98, 0.97, f"RMSE {err[len(err) // 2]:.2f} BW, {np.mean(np.abs(y[i] - mu[i]) <= 1.96 * sd[i]) * 100:.0f}% inside", transform=ax.transAxes, ha="right", va="top", fontsize=9)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{labels[c]} [BW]")
            if j == 2:
                ax.set_xlabel("Gait cycle [%]")
    axes[0][0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


precision_path = os.path.join(args.out, f"trivariate_precision_per_block_{tag}_t{t_show:03d}.png")
plot_precision(precision_show, count_show, N, args.block_size, precision_path, t_show)
curves_path = os.path.join(args.out, f"trivariate_predicted_curves_{tag}.png")
plot_curves(curves_path)
print(f"\nfigures saved to {precision_path} and {curves_path}")
