#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
import cv2

# ==========================================================
# CONFIG
# ==========================================================
if len(sys.argv) < 2:
    print("Usage: python multiview_fuse_da3.py <tour_id>")
    sys.exit(1)

tour_id = sys.argv[1]
DATASET_DIR = Path("/mnt/shared/data") / tour_id
CUBE_IMG_DIR = DATASET_DIR / "undistorted" / "images"
CUBE_DEPTH_DIR = DATASET_DIR / "undistorted" / "undistort_depth_output"
POSE_DIR = DATASET_DIR /"multiview_fused" / "rot_trans_matrix_npy"

OUT_PLY = DATASET_DIR / "da3_multiview_fused_enu.ply"
MERGED_PLY = DATASET_DIR / "undistorted" / "depthmaps" / "merged.ply"
OUT_DEBUG_DIR = DATASET_DIR / "multiview_fused" / "mv_debug_projections"

STRIDE = 4
MIN_DEPTH = 0.1
MAX_DEPTH = 150.0
FACES = ["front", "back", "left", "right", "top", "bottom"]
# ==========================================================

def get_face_rotation(face_name):
    if face_name == 'front':   return np.eye(3)
    if face_name == 'back':    return np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    if face_name == 'right':   return np.array([[0,0,1],[0,1,0],[-1,0,0]])
    if face_name == 'left':    return np.array([[0,0,-1],[0,1,0],[1,0,0]])
    if face_name == 'top':     return np.array([[1,0,0],[0,0,1],[0,-1,0]])
    if face_name == 'bottom':  return np.array([[1,0,0],[0,0,-1],[0,1,0]])
    raise ValueError(f"Unknown face: {face_name}")

def load_pose(base_id):
    npz_path = POSE_DIR / f"shot_{base_id}.jpg.npz"
    data = np.load(npz_path)
    return data["rotation"], data["centre"]

def backproject_cube_face(img_path, depth_path, R_global, C_global, face_name):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    depth = np.load(depth_path)["depth"].astype(np.float32)

    if depth.shape != (H, W):
        return np.zeros((0,3)), np.zeros((0,3))

    u = np.arange(0, W, STRIDE)
    v = np.arange(0, H, STRIDE)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()

    d = depth[vv, uu]
    valid = (d > MIN_DEPTH) & (d < MAX_DEPTH)
    if not np.any(valid):
        return np.zeros((0,3)), np.zeros((0,3))

    uu, vv, d = uu[valid], vv[valid], d[valid]
    cols = img_np[vv, uu] / 255.0

    x = (uu - W/2) / (W/2)
    y = (vv - H/2) / (H/2)
    dirs_local = np.stack([x, y, np.ones_like(x)], axis=1)
    dirs_local /= np.linalg.norm(dirs_local, axis=1, keepdims=True)

    R_face = get_face_rotation(face_name)
    dirs_cam = (R_face @ dirs_local.T).T
    pts_cam = dirs_cam * d[:, None]
    pts_world = (R_global.T @ pts_cam.T).T + C_global.reshape(1,3)
    return pts_world.astype(np.float32), cols.astype(np.float32)

def project_world_to_face(pts_world, R_global, C_global, face_name, H, W):
    """Project 3D world points into cube-face pixel coords."""
    R_face = get_face_rotation(face_name)
    pts_cam = (R_global @ (pts_world - C_global).T).T
    pts_face = (R_face.T @ pts_cam.T).T
    x, y, z = pts_face[:,0], pts_face[:,1], pts_face[:,2]
    valid = z > 0
    if not np.any(valid):
        return np.array([]), np.array([])
    x, y, z = x[valid], y[valid], z[valid]
    u = (x / z * 0.5 + 0.5) * W
    v = (y / z * 0.5 + 0.5) * H
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[mask].astype(np.int32), v[mask].astype(np.int32)

def main():
    print("\n[MV] Starting multiview fusion (cube faces)...")
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Fuse cube faces (existing behavior)
    pts_all, cols_all = [], []
    all_faces = sorted(CUBE_IMG_DIR.glob("*.jpg"))
    base_ids = sorted(set(p.name.split(".jpg_perspective_view_")[0] for p in all_faces))

    for base_id in base_ids:
        try:
            R_global, C_global = load_pose(base_id)
        except Exception as e:
            print(f"[WARN] Missing pose for {base_id}: {e}")
            continue

        for face in FACES:
            face_pattern = f"{base_id}.jpg_perspective_view_{face}.jpg"
            img_path = CUBE_IMG_DIR / face_pattern
            depth_path = CUBE_DEPTH_DIR / f"{face_pattern}_depth_meters.npz"
            if not img_path.exists() or not depth_path.exists():
                continue
            pts, cols = backproject_cube_face(img_path, depth_path, R_global, C_global, face)
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
    
    print("[MV] Creating debug projections with merged.ply comparison...")
    gen_cloud = o3d.io.read_point_cloud(str(OUT_PLY))
    merged_cloud = None
    if MERGED_PLY.exists():
        merged_cloud = o3d.io.read_point_cloud(str(MERGED_PLY))
        print("[MV] Loaded reference merged.ply for overlay.")
    else:
        print("[WARN] Reference merged.ply not found, skipping overlay.")

    for base_id in base_ids:
        R_global, C_global = load_pose(base_id)
        for face in FACES:
            face_pattern = f"{base_id}.jpg_perspective_view_{face}.jpg"
            img_path = CUBE_IMG_DIR / face_pattern
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None: 
                continue
            H, W = img.shape[:2]
            vis = img.copy()

            # Project generated points
            u, v = project_world_to_face(np.asarray(gen_cloud.points), R_global, C_global, face, H, W)
            vis[v, u] = (0, 0, 255)  # red for generated fusion

            # Project reference points
            if merged_cloud is not None:
                um, vm = project_world_to_face(np.asarray(merged_cloud.points), R_global, C_global, face, H, W)
                vis[vm, um] = (0, 255, 0)  # green for merged.ply

            out_path = OUT_DEBUG_DIR / f"{base_id}_{face}_overlay.png"
            cv2.imwrite(str(out_path), vis)
            print(f"[MV] Saved overlay projection: {out_path}")

    print("[MV] Debug projections complete.")

if __name__ == "__main__":
    main()
