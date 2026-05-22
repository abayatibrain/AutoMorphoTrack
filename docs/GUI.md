# AutoMorphoTrack GUI (napari plugin)

AutoMorphoTrack ships a [napari](https://napari.org) plugin so the whole
pipeline can be driven by a point-and-click interface — no Python required.
This is the option for users who previously preferred FIJI/ImageJ but want
AutoMorphoTrack's organelle-pair workflow.

## Install

```bash
pip install "automorphotrack[napari]"
```

The `[napari]` extra installs `napari`, `magicgui`, and `qtpy`. On Linux
and Windows it will also pull `PyQt5` (the Qt binding); on macOS Apple
silicon you may need to install Qt separately, see
[napari's install docs](https://napari.org/stable/tutorials/fundamentals/installation.html).

## Launch

Two equivalent ways:

```bash
# 1. Launch napari with all AMT widgets pre-docked
automorphotrack gui

# 2. Open plain napari and find AMT in the Plugins menu
napari
# → Plugins → AutoMorphoTrack → <pick a widget>
```

## Widgets

| Widget | What it does | Inputs | Output |
|---|---|---|---|
| **Detect organelles** | Live preview of segmentation on frame 0 | TIF path, channels, backend, thr_factor, min_size | Adds raw + mask layers to viewer |
| **Shape features** | Per-organelle morphometrics across all frames | TIF path, output dir | CSV + violin/KDE PNG |
| **Tracking + motility** | Trajectories + velocity / displacement | TIF path, output dir, min_detectable_displacement, fps | CSVs + scatter PNG with annotated persistent-motion diagonal |
| **Colocalization** | Manders / Jaccard / Pearson / cosine | TIF path, output dir | CSV + metric-definitions CSV + overlay video |
| **Validation sweep** | Real parameter sweep with Dice/IoU/F1 | TIF path, sweep param, values | CSV + sensitivity PNG |
| **Run full pipeline** | All stages into one folder | TIF path, output root | Subfolder per stage |

## Workflow

1. Open napari (`automorphotrack gui`).
2. Drag a `.tif` stack into the viewer or open it via File → Open.
3. From the docked **Detect** widget, pick the segmentation backend
   (`otsu`, `sauvola`, `niblack`, `local_otsu`, `subtracted`) and tweak
   `thr_factor` / `min_size` until the mask overlay looks right on frame 0.
4. From **Shape features** / **Tracking + motility** / **Colocalization**
   widgets, point each at the same TIF and an output folder.
5. Click **Run full pipeline** when satisfied, or run stages individually
   to keep their outputs separate.

Every widget calls the same Python functions documented in the [package
README](../README.md), so anything you can do in the GUI is also a
one-line script.

## Verifying it works headlessly

The plugin module imports cleanly even when napari isn't installed, so
`pip install automorphotrack` (without the `[napari]` extra) stays light.
CI confirms the widgets register correctly with napari's plugin manager:

```bash
python -c "from automorphotrack import napari_plugin; print(napari_plugin._NAPARI_AVAILABLE)"
```

## Relationship to the MCP connector

The napari plugin is for users who want to *see* what's happening at each
step. The [MCP connector](MCP.md) is for users who want Claude Code to
run analyses on their behalf from natural-language prompts. Both wrap the
same Python API, so they can be mixed: use Claude Code to script a batch,
then open napari to inspect one stack's results visually.
