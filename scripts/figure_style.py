"""Shared, English-only print typography and method encoding for evidence plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("basic", "agent")
DISPLAY_NAMES = {"basic": "Basic", "agent": "IntALNS"}
COLORS = {"agent": "#287c65", "basic": "#be6258", "tie": "#e5e5e5"}
STYLE = {
    "basic": dict(color=COLORS["basic"], linestyle="--", marker="s", label=DISPLAY_NAMES["basic"]),
    "agent": dict(color=COLORS["agent"], linestyle="-", marker="o", label=DISPLAY_NAMES["agent"]),
}


def configure():
    # STIX is distributed with Matplotlib: rebuilding needs no machine-local font.
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelcolor": "black", "axes.edgecolor": "#555555",
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "stix", "savefig.facecolor": "white",
    })


def export(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf",):
        fig.savefig(path.with_suffix("." + extension), dpi=600)
    plt.close(fig)
