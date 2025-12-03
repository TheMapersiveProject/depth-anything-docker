#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
import cv2

# ========================= CONFIG =========================
DATASET_DIR = Path("/mnt/shared/data/576258")

IMAGES_DIR  = DATASET_DIR / "images"
DEPTH_DIR   = DATASET_DIR / "depth_output"
POSE_DIR    = DATASET_DIR / "rot_trans_matrix_npy"

OUT_PLY     = DATASET_DIR / "da3_multiview_fused_enu.ply"
OUT_DEBUG_DIR = DATASET_DIR / "mv_debug_projections"

STRIDE = 4
MIN_DEPTH = 0.1
MAX_DEPTH = 200.0
# ==========================================================


def load_pose(stem):
    """Load R (world→cam) and C (camera center ENU)."""
    npz_path = POSE_DIR / f"shot_{stem}.jpg.npz"
    data = np.load(npz_path)
    R = data["rotation"]
    C = data["centre"]
    return R, C


# ==========================================================
# 1) BACKPROJECT: Equirectangular → ENU point cloud
# ==========================================================
def backproject_equirect(img_path, depth_path, R, C):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    W, H = img.size

    depth = np.load(depth_path)["depth"].astype(np.float32)
    if depth.shape != (H, W):
        return np.zeros((0,3)), np.zeros((0,3))

    # Pixel sampling
    u = np.arange(0, W, STRIDE)
    v = np.arange(0, H, STRIDE)
    uu, vv = np.meshgrid(u, v)
    uu = uu.ravel()
    vv = vv.ravel()

    d = depth[vv, uu]
    valid = (d > MIN_DEPTH) & (d < MAX_DEPTH)

    if not np.any(valid):
        return np.zeros((0,3)), np.zeros((0,3))

    uu = uu[valid]
    vv = vv[valid]
    d  = d[valid]
    cols = img_np[vv, uu] / 255.0

    # Convert to spherical angles (EXACT inverse of your projection script)
    theta = (uu / W) * 2*np.pi - np.pi          # [-π, π]
    phi   = (vv / H) * np.pi                    # [0, π]

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    sint = np.sin(theta)
    cost = np.cos(theta)

    # Camera ray (x right, y down, z forward)
    dirs = np.stack([
        sinphi * sint,
        -cosphi,
        sinphi * cost
    ], axis=1)

    pts_cam = dirs * d[:, None]
    pts_world = (R.T @ pts_cam.T).T + C.reshape(1,3)

    return pts_world.astype(np.float32), cols.astype(np.float32)


# ==========================================================
# 2) FOR DEBUG: Project fused PCD back into each image
# ==========================================================
def project_points_to_image(pts_world, R, C, W, H):
    """
    world (ENU) -> camera -> equirectangular pixel coords
    EXACT inverse of equirectangular backprojection.
    """
    pts_cam = (R @ (pts_world - C).T).T
    x, y, z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    r = np.linalg.norm(pts_cam, axis=1)

    # spherical
    theta = np.arctan2(x, z)          # [-π, π]
    phi = np.arccos(np.clip(-y/r, -1, 1))  # [0, π]

    # to pixel
    u = (theta + np.pi) / (2*np.pi) * W
    v = (phi / np.pi) * H

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(u) & np.isfinite(v)
    return u[valid].astype(np.int32), v[valid].astype(np.int32), valid


# ==========================================================
# 3) MAIN MULTIVIEW PIPELINE
# ==========================================================
def main():
    print("\n[MV] Starting multiview fusion...")
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    pts_all = []
    cols_all = []

    images = sorted(IMAGES_DIR.glob("*.jpg"))

    for img_path in images:
        stem = img_path.stem
        depth_path = DEPTH_DIR / f"{stem}_depth_meters.npz"
        if not depth_path.exists():
            print(f"[WARN] Missing depth for {stem}")
            continue

        R, C = load_pose(stem)
        pts, cols = backproject_equirect(img_path, depth_path, R, C)

        if pts.shape[0] == 0:
            continue

        pts_all.append(pts)
        cols_all.append(cols)

    if not pts_all:
        print("[ERR] No points produced.")
        return

    pts_all = np.vstack(pts_all)
    cols_all = np.vstack(cols_all)

    # Save fused cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_all)
    pc.colors = o3d.utility.Vector3dVector(cols_all)
    o3d.io.write_point_cloud(str(OUT_PLY), pc)
    print("[MV] Saved fused cloud:", OUT_PLY)

    # ======================================================
    # DEBUG: Project fused PCD back into each image
    # ======================================================
    print("[MV] Creating debug projections...")

    pts_world = pts_all

    for img_path in images:
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        stem = img_path.stem

        R, C = load_pose(stem)
        u, v, _ = project_points_to_image(pts_world, R, C, W, H)

        vis = img.copy()
        vis[v, u] = (0, 0, 255)  # red projection points

        out_debug = OUT_DEBUG_DIR / f"{stem}_projection_debug.png"
        cv2.imwrite(str(out_debug), vis)
        print(f"[MV] Projection saved: {out_debug}")


if __name__ == "__main__":
    main()
