"""Helpers for the OpenSim JCF data: loading, wide-to-curve reshaping, overlay plots.

Used by codes/jcf-gp-colab.ipynb after cloning the repository:
    import sys; sys.path.insert(0, "OpenSim-JCF-GP/codes")
    from jcfgp import *
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SPEED_COLOURS = {1.0: "#0072B2", 1.25: "#D55E00", 1.4: "#000000", 1.5: "#009E73"}   # Okabe-Ito, one colour per belt speed
JOINT_COLOURS = {"hip": "#CC79A7", "knee": "#0072B2", "ankle": "#009E73"}
GRID = np.arange(101)                                                                # percent of the gait cycle
TOE_OFF = 62                                                                         # approximate end of stance, percent
INPUT_CURVES = ["grf_vertical_BW", "grf_anteroposterior_BW", "grf_mediolateral_BW",
                "hip_flexion_deg", "hip_adduction_deg", "hip_rotation_deg", "knee_flexion_deg", "ankle_angle_deg"]
OUTPUT_CURVES = ["hip_jcf_BW", "knee_jcf_BW", "ankle_jcf_BW"]
SCALARS = ["belt_speed_mps", "mass_kg", "height_m"]
LABELS = {"grf_vertical_BW": "Vertical GRF [BW]", "grf_anteroposterior_BW": "Fore-aft GRF [BW]", "grf_mediolateral_BW": "Medio-lateral GRF [BW]",
          "hip_flexion_deg": "Hip flexion [deg]", "hip_adduction_deg": "Hip adduction [deg]", "hip_rotation_deg": "Hip rotation [deg]",
          "knee_flexion_deg": "Knee flexion [deg]", "ankle_angle_deg": "Ankle dorsiflexion [deg]",
          "hip_jcf_BW": "Hip JCF [BW]", "knee_jcf_BW": "Knee JCF [BW]", "ankle_jcf_BW": "Ankle JCF [BW]"}


def load_data(data_dir):
    """Returns x_train, y_train, x_test, y_test as data frames and checks that x and y rows are the same strides."""
    x_train = pd.read_csv(f"{data_dir}/x-train.csv"); y_train = pd.read_csv(f"{data_dir}/y-train.csv")
    x_test = pd.read_csv(f"{data_dir}/x-test.csv"); y_test = pd.read_csv(f"{data_dir}/y-test.csv")
    ids = ["subject", "block", "stride_in_block"]
    assert (x_train[ids].values == y_train[ids].values).all() and (x_test[ids].values == y_test[ids].values).all(), "x and y rows are not aligned"
    return x_train, y_train, x_test, y_test


def curve(df, name):
    """The curve `name` of every row as a (strides, 101) array, from the columns name_t000 ... name_t100."""
    return df[[f"{name}_t{t:03d}" for t in GRID]].to_numpy()


def overlay(ax, df, name, label=None):
    """All strides of one curve on `ax`, coloured by belt speed, with the mean curve of each speed on top."""
    C = curve(df, name); speeds = df["belt_speed_mps"].to_numpy()
    for v in sorted(SPEED_COLOURS):
        rows = speeds == v
        if rows.any():
            ax.plot(GRID, C[rows].T, color=SPEED_COLOURS[v], lw=0.5, alpha=0.25)
    for v in sorted(SPEED_COLOURS):
        rows = speeds == v
        if rows.any():
            ax.plot(GRID, C[rows].mean(axis=0), color=SPEED_COLOURS[v], lw=2.5, label=f"{v:g} m/s (n={rows.sum()})")
    ax.axvline(TOE_OFF, color="grey", lw=0.8, ls=":")
    ax.set_title(label or LABELS.get(name, name), loc="left"); ax.set_xlim(0, 100); ax.grid(alpha=0.3)


def overlay_grid(df, names, ncols=4, title=None, figsize=None):
    """One overlay panel per curve name, as a grid; returns the figure."""
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 4 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        overlay(ax, df, name)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Gait cycle [%]")
    axes[0][0].legend(frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def show_stride(x, y, i):
    """Inputs (8 panels) and outputs (one panel) of training stride i; returns both figures."""
    row = x.iloc[i]
    fig1, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, name in zip(axes.flat, INPUT_CURVES):
        ax.plot(GRID, curve(x, name)[i], color="#0b0b0b", lw=2)
        ax.set_title(LABELS[name], loc="left"); ax.set_xlim(0, 100); ax.grid(alpha=0.3)
    fig1.suptitle(f"Inputs of one stride: {row['subject']}, {row['block']}, stride {row['stride_in_block']}, "
                  f"{row['belt_speed_mps']:g} m/s, {row['mass_kg']} kg, {row['height_m']} m", x=0.01, ha="left")
    fig1.tight_layout()
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    for name in OUTPUT_CURVES:
        j = name.split("_")[0]; ax.plot(GRID, curve(y, name)[i], color=JOINT_COLOURS[j], lw=2.5, label=j)
    ax.set_title("Outputs of the same stride: joint contact forces", loc="left"); ax.set_xlabel("Gait cycle [%]"); ax.set_ylabel("JCF [BW]")
    ax.set_xlim(0, 100); ax.grid(alpha=0.3); ax.legend(frameon=False); fig2.tight_layout()
    return fig1, fig2


def per_subject(x, y, name="knee_jcf_BW", speed=1.0, ncols=4):
    """One panel per subject: that subject's strides of `name` at one belt speed, with the grand mean dashed."""
    C = curve(y, name); subjects = sorted(x["subject"].unique()); nrows = -(-len(subjects) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, subjects):
        rows = ((x["subject"] == s) & (x["belt_speed_mps"] == speed)).to_numpy()
        ax.plot(GRID, C[rows].T, color=JOINT_COLOURS.get(name.split("_")[0], "#0b0b0b"), lw=1, alpha=0.7)
        ax.plot(GRID, C.mean(axis=0), color="grey", lw=1.5, ls="--")
        ax.set_title(s.replace("NIH_ACL_", ""), loc="left"); ax.grid(alpha=0.3)
    for ax in axes.flat[len(subjects):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Gait cycle [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel(LABELS.get(name, name))
    fig.suptitle(f"{LABELS.get(name, name)} at {speed:g} m/s, one panel per subject (dashed grey: mean of all strides)", x=0.01, ha="left")
    fig.tight_layout()
    return fig
