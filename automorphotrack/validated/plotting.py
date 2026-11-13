"""Optional blue/orange figures with bounded empirical distributions."""

import numpy as np


BLUE = "#0072B2"
ORANGE = "#E69F00"


def render_outputs(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stack = results["stack"]
    config = results["config"]
    rgb = np.zeros((*stack.shape[1:3], 3), float)
    for channel, color in ((config.mito_channel, (0, 0.45, 0.7)), (config.lyso_channel, (0.9, 0.6, 0))):
        image = stack[0, ..., channel].astype(float)
        normalized = (image - image.min()) / np.ptp(image) if np.ptp(image) else np.zeros_like(image)
        rgb += normalized[..., None] * np.array(color)
    overlap = results["mito_masks"][0] & results["lyso_masks"][0]
    rgb = np.clip(rgb, 0, 1)
    plt.imsave(out_dir / "Frame0_BlueOrange.png", rgb)
    rgb[overlap] = 1
    plt.imsave(out_dir / "Frame0_OverlapWhite.png", rgb)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color in (("mito", BLUE), ("lyso", ORANGE)):
        shapes = results[name + "_shapes"]
        motility = results[name + "_motility"]
        for ax, values, xlabel in ((axes[0], shapes["Circularity"], "Contour circularity"),
                                   (axes[1], motility["Mean_Speed"], "Mean speed")):
            values = values.dropna().sort_values().to_numpy()
            if len(values):
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=name, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Empirical cumulative fraction")
            ax.set_ylim(0, 1)
    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(left=0)
    axes[1].set_xlabel("Mean speed (" + ("um" if config.pixel_size_um else "px") + "/" +
                       ("s" if config.frame_interval_s else "frame") + ")")
    for ax in axes:
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "Shape_Motility_ECDF.png", dpi=200)
    plt.close(fig)
    corr = results["correlation"]
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=6)
    ax.set_title("Framewise Spearman coefficients: descriptive, dependent frames")
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(out_dir / "Frame_Descriptive_Spearman.png", dpi=200)
    plt.close(fig)
