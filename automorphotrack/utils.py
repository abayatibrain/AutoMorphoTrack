# ============================================================
# AutoMorphoTrack Utilities
# ============================================================

import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_high_dpi(fig, path, dpi=600):
    """Save Matplotlib figure with high resolution."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")

def upscale_frame(img, scale=4):
    """Upscale an image for high-quality visualization."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def write_video(frames, path, fps=5):
    """Write RGB frames into an MP4 video."""
    if not frames:
        print("No frames provided for video.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for fr in frames:
        out.write(cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved video: {path}")


# ============================================================
# Statistics annotation helper (reviewer R2.5.iv)
# ============================================================
def annotate_stats(ax, group_a, group_b, test="auto", group_a_name="A",
                   group_b_name="B", y=None, alpha=0.05, decimals=3):
    """Print N, test, p-value and effect size on a matplotlib axis.

    Reviewer #2 asked for explicit reporting of sample size, statistical test
    used, p-value, and effect size on every comparison figure. Use this helper
    on any axis where a comparison is being shown::

        annotate_stats(ax, mito["Area"], lyso["Area"],
                       group_a_name="Mito", group_b_name="Lyso")

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to annotate.
    group_a, group_b : array-like
        Numeric samples.
    test : {"auto","welch","ttest","mannwhitney"}, default "auto"
        Statistical test. "auto" chooses Welch's t-test when both groups pass
        Shapiro–Wilk normality at ``alpha`` else Mann–Whitney U.
    group_a_name, group_b_name : str
        Labels used in the annotation.
    y : float, optional
        Vertical placement (axis fraction). Defaults to top-right of the axis.
    alpha : float, default 0.05
        Normality threshold for ``test="auto"``.
    decimals : int, default 3
        Decimal places for the reported statistic.

    Returns
    -------
    dict
        ``{"test": ..., "p": ..., "effect_size": ..., "n_a": ..., "n_b": ...}``
    """
    import numpy as np
    from scipy import stats

    a = np.asarray(group_a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(group_b, dtype=float)
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 3 or n_b < 3:
        text = f"N={n_a} vs {n_b} (too few for test)"
        ax.text(0.98, y if y is not None else 0.97, text, ha='right', va='top',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        return {"test": None, "p": None, "effect_size": None,
                "n_a": n_a, "n_b": n_b}

    chosen = test
    if test == "auto":
        try:
            _, pa = stats.shapiro(a if n_a <= 5000 else np.random.choice(a, 5000, replace=False))
            _, pb = stats.shapiro(b if n_b <= 5000 else np.random.choice(b, 5000, replace=False))
            chosen = "welch" if (pa > alpha and pb > alpha) else "mannwhitney"
        except Exception:
            chosen = "mannwhitney"

    if chosen == "welch":
        stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        eff = (np.mean(a) - np.mean(b)) / (pooled + 1e-12)
        eff_label = "d"
    elif chosen == "ttest":
        stat, p = stats.ttest_ind(a, b, nan_policy="omit")
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        eff = (np.mean(a) - np.mean(b)) / (pooled + 1e-12)
        eff_label = "d"
    elif chosen == "mannwhitney":
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial r
        eff = 1 - (2 * u) / (n_a * n_b)
        eff_label = "r"
    else:
        raise ValueError(f"unknown test {test}")

    text = (f"N({group_a_name})={n_a}, N({group_b_name})={n_b}\n"
            f"{chosen}: p={p:.{decimals}g}\n"
            f"{eff_label}={eff:.{decimals}f}")
    ax.text(0.98, y if y is not None else 0.97, text, ha='right', va='top',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='gray'))
    return {"test": chosen, "p": float(p), "effect_size": float(eff),
            "n_a": n_a, "n_b": n_b}
