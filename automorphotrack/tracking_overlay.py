# ============================================================
# AutoMorphoTrack – Organelle Tracking with Real-Intensity Overlays
# ============================================================

import numpy as np, cv2, tifffile
from pathlib import Path
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
from automorphotrack.utils import ensure_dir
from automorphotrack.shape_features import detect_mask

def track_overlay(
    tif_path="Composite.tif",
    out_dir="TrackingOverlay_Outputs",
    mito_channel=0,
    lyso_channel=1,
    fps=10,
    upscale=4.0,
    line_thickness=1):

    ensure_dir(out_dir)
    stack = tifffile.imread(tif_path)
    if stack.ndim == 3:
        stack = stack[..., np.newaxis]
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for tracking overlays")

    # ---------- Helper functions ----------
    def get_centroids(mask):
        lbl = label(mask)
        return np.array([p.centroid for p in regionprops(lbl)]) if lbl.max() > 0 else np.empty((0, 2))

    def nearest_neighbor(prev, curr, max_dist=15):
        if len(prev) == 0 or len(curr) == 0:
            return {}
        tree = cKDTree(curr)
        dists, idx = tree.query(prev, distance_upper_bound=max_dist)
        return {i: j for i, j in enumerate(idx) if j < len(curr)}

    def upscale_frame(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)

    def write_video(frames, out_path, fps):
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            out.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        out.release()

    # ---------- Extract centroid tracks ----------
    mito_tracks, lyso_tracks = [], []
    for f in range(n_frames):
        mito_mask, _ = detect_mask(stack[f], mito_channel, min_size=2, thr_factor=0.65)
        lyso_mask, _ = detect_mask(stack[f], lyso_channel, min_size=2, thr_factor=0.85)
        mito_tracks.append(get_centroids(mito_mask))
        lyso_tracks.append(get_centroids(lyso_mask))

    # ---------- Build cumulative trajectory frames ----------
    mito_paths, lyso_paths = [], []
    mito_frames, lyso_frames, composite_frames = [], [], []

    for f in range(n_frames):
        mito_mask, _ = detect_mask(stack[f], mito_channel, min_size=2, thr_factor=0.65)
        lyso_mask, _ = detect_mask(stack[f], lyso_channel, min_size=2, thr_factor=0.85)

        mito_img = stack[f, :, :, mito_channel]
        lyso_img = stack[f, :, :, lyso_channel]

        mito_norm = cv2.normalize(mito_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lyso_norm = cv2.normalize(lyso_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        mito_base = cv2.cvtColor(mito_norm, cv2.COLOR_GRAY2RGB)
        lyso_base = cv2.cvtColor(lyso_norm, cv2.COLOR_GRAY2RGB)
        comp_merge = np.clip(mito_norm * 0.5 + lyso_norm * 0.5, 0, 255).astype(np.uint8)
        comp_base = cv2.cvtColor(comp_merge, cv2.COLOR_GRAY2RGB)

        if f > 0:
            prev_m, curr_m = mito_tracks[f - 1], mito_tracks[f]
            prev_l, curr_l = lyso_tracks[f - 1], lyso_tracks[f]
            if len(prev_m) and len(curr_m):
                nn = nearest_neighbor(prev_m, curr_m)
                for k, v in nn.items():
                    mito_paths.append((prev_m[k], curr_m[v]))
            if len(prev_l) and len(curr_l):
                nn = nearest_neighbor(prev_l, curr_l)
                for k, v in nn.items():
                    lyso_paths.append((prev_l[k], curr_l[v]))

        mito_frame = mito_base.copy()
        lyso_frame = lyso_base.copy()
        comp_frame = comp_base.copy()

        for p1, p2 in mito_paths:
            cv2.line(mito_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])),
                     (1, 115, 178), line_thickness, cv2.LINE_AA)
            cv2.line(comp_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])),
                     (1, 115, 178), line_thickness, cv2.LINE_AA)
        for p1, p2 in lyso_paths:
            cv2.line(lyso_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])),
                     (222, 143, 5), line_thickness, cv2.LINE_AA)
            cv2.line(comp_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])),
                     (222, 143, 5), line_thickness, cv2.LINE_AA)

        mito_frames.append(upscale_frame(mito_frame))
        lyso_frames.append(upscale_frame(lyso_frame))
        composite_frames.append(upscale_frame(comp_frame))

    # ---------- Save outputs ----------
    write_video(mito_frames, Path(out_dir) / "Mito_CumulativeTracks.mp4", fps=fps)
    write_video(lyso_frames, Path(out_dir) / "Lyso_CumulativeTracks.mp4", fps=fps)
    write_video(composite_frames, Path(out_dir) / "Composite_CumulativeTracks.mp4", fps=fps)

    for name, frames in zip(["Mito", "Lyso", "Composite"],
                             [mito_frames, lyso_frames, composite_frames]):
        out_path = Path(out_dir) / f"Cumulative_{name}.png"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"Tracking overlay complete — results saved in {Path(out_dir).resolve()}")
