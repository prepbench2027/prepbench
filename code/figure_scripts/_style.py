"""Shared plotting style for PrepBench figures.

Import `apply_style()` at the top of each figure script. Kept deliberately
minimal and dependency-free (matplotlib only) so every figure script is
standalone and reproducible.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config display order and human labels used across figures.
CONFIG_ORDER = [
    "Flat-Rand", "D-Rand",
    "L-S-F-A", "B-S-F-A", "D-S-F-A",
    "D-C-F-A", "D-S-F-B", "D-S-T-A", "D-C-T-B",
]

# Group colors: random controls vs structured/cost-aware vs simple-structure.
RANDOM_CONTROLS = {"Flat-Rand", "D-Rand"}
def config_color(cfg):
    if cfg in RANDOM_CONTROLS:
        return "#1b7837"      # green: random controls
    if cfg in {"L-S-F-A", "B-S-F-A"}:
        return "#6a51a3"      # purple: simple structures
    return "#b2182b"          # red: structured / cost-aware / staged


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,   # editable text in the PDF
        "ps.fonttype": 42,
    })
