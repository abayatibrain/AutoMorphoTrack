# ============================================================
# AutoMorphoTrack – Organelle Tracking (Cumulative Paths)
# ============================================================

import numpy as np, cv2, pandas as pd, tifffile
from pathlib import Path
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from automorphotrack.utils import ensure_dir, write_video, upscale_frame
from automorphotrack.shape_features import detect_mask

def track_organelles(
    tif_path="Composite.tif",
    out_dir="Tracking_Outputs",
    mito_channel=0,
    lyso_channel=1,
    fps=5,
    max_dist=15):

    ensure_dir(out_dir)
    stack = tifffile.imread(tif_path)
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for organelle tracking")

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

    mito_tracks, lyso_tracks = [], []
    mito_data, lyso_data = [], []

    # ---------- Collect centroids per frame ----------
    for f in range(n_frames):
        mito_mask, _ = detect_mask(stack[f], mito_channel, min_size=2, thr_factor=0.65)
        lyso_mask, _ = detect_mask(stack[f], lyso_channel, min_size=2, thr_factor=0.85)

        mito_curr = get_centroids(mito_mask)
        lyso_curr = get_centroids(lyso_mask)
        mito_tracks.append(mito_curr)
        lyso_tracks.append(lyso_curr)

        for i, (y, x) in enumerate(mito_curr):
            mito_data.append([f, i, x, y])
        for i, (y, x) in enumerate(lyso_curr):
            lyso_data.append([f, i, x, y])

    # ---------- Save centroid coordinates ----------
    df_mito = pd.DataFrame(mito_data, columns=["Frame", "Organelle", "X", "Y"])
    df_lyso = pd.DataFrame(lyso_data, columns=["Frame", "Organelle", "X", "Y"])
    df_mito.to_csv(Path(out_dir) / "Mito_Tracks.csv", index=False)
    df_lyso.to_csv(Path(out_dir) / "Lyso_Tracks.csv", index=False)

    # ---------- Compute displacement and velocity ----------
    def compute_disp_vel(df):
        disps, vels = [], []
        for oid in df["Organelle"].unique():
            sub = df[df["Organelle"] == oid].sort_values("Frame")
            if len(sub) > 1:
                dx, dy = np.diff(sub["X"]), np.diff(sub["Y"])
                disp = np.sqrt(dx ** 2 + dy ** 2)
                disps.append([oid, disp.sum()])
                vels.append([oid, disp.mean() * fps])
        return (pd.DataFrame(disps, columns=["Organelle", "Total_Displacement"]),
                pd.DataFrame(vels, columns=["Organelle", "Mean_Velocity"]))

    mito_disp, mito_vel = compute_disp_vel(df_mito)
    lyso_disp, lyso_vel = compute_disp_vel(df_lyso)

    mito_disp.to_csv(Path(out_dir) / "Mito_Displacement.csv", index=False)
    mito_vel.to_csv(Path(out_dir) / "Mito_Velocity.csv", index=False)
    lyso_disp.to_csv(Path(out_dir) / "Lyso_Displacement.csv", index=False)
    lyso_vel.to_csv(Path(out_dir) / "Lyso_Velocity.csv", index=False)

    # ---------- Build cumulative frames ----------
    mito_paths, lyso_paths = [], []
    mito_frames, lyso_frames, composite_frames = [], [], []

    for f in range(n_frames):
        frame_rgb = np.zeros_like(stack[f][..., :3])
        if f > 0:
            if len(mito_tracks[f - 1]) and len(mito_tracks[f]):
                for k, v in nearest_neighbor(mito_tracks[f - 1], mito_tracks[f], max_dist).items():
                    mito_paths.append((mito_tracks[f - 1][k], mito_tracks[f][v]))
            if len(lyso_tracks[f - 1]) and len(lyso_tracks[f]):
                for k, v in nearest_neighbor(lyso_tracks[f - 1], lyso_tracks[f], max_dist).items():
                    lyso_paths.append((lyso_tracks[f - 1][k], lyso_tracks[f][v]))

        mito_frame, lyso_frame, comp_frame = frame_rgb.copy(), frame_rgb.copy(), frame_rgb.copy()

        for p1, p2 in mito_paths:
            cv2.line(mito_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])), (1, 115, 178), 1)
            cv2.line(comp_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])), (1, 115, 178), 1)
        for p1, p2 in lyso_paths:
            cv2.line(lyso_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])), (222, 143, 5), 1)
            cv2.line(comp_frame, tuple(np.int32(p1[::-1])), tuple(np.int32(p2[::-1])), (222, 143, 5), 1)

        mito_frames.append(upscale_frame(mito_frame))
        lyso_frames.append(upscale_frame(lyso_frame))
        composite_frames.append(upscale_frame(comp_frame))

    # ---------- Save videos ----------
    write_video(mito_frames, Path(out_dir) / "Mito_CumulativeTracks.mp4", fps=fps)
    write_video(lyso_frames, Path(out_dir) / "Lyso_CumulativeTracks.mp4", fps=fps)
    write_video(composite_frames, Path(out_dir) / "Composite_CumulativeTracks.mp4", fps=fps)

    # ---------- Save final stills ----------
    cv2.imwrite(str(Path(out_dir) / "Cumulative_Mito.png"), cv2.cvtColor(mito_frames[-1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(out_dir) / "Cumulative_Lyso.png"), cv2.cvtColor(lyso_frames[-1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(out_dir) / "Cumulative_Composite.png"), cv2.cvtColor(composite_frames[-1], cv2.COLOR_RGB2BGR))

    print(f"Tracking complete — outputs saved in {Path(out_dir).resolve()}")
