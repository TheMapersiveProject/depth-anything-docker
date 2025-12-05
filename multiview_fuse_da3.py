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

# --- Auto-fix if running inside /multiview_fused ---
if not (DATASET_DIR / "undistorted").exists():
    DATASET_DIR = DATASET_DIR.parent / tour_id
    print(f"[INFO] Adjusted DATASET_DIR → {DATASET_DIR}")

CUBE_IMG_DIR = DATASET_DIR / "undistorted" / "undistort_depth_output"
CUBE_DEPTH_DIR = CUBE_IMG_DIR

POSE_DIR = DATASET_DIR / "rot_trans_matrix_npy"
if not POSE_DIR.exists():
    POSE_DIR = DATASET_DIR.parent / "rot_trans_matrix_npy"

OUT_PLY = DATASET_DIR / "multiview_fused" / "da3_multiview_fused_enu.ply"
MERGED_PLY = DATASET_DIR / "undistorted" / "depthmaps" / "merged.ply"
OUT_DEBUG_DIR = DATASET_DIR / "multiview_fused" / "mv_debug_projections"

print("cube depth path =",CUBE_DEPTH_DIR)
print("cube faces path =",CUBE_IMG_DIR)
print("pose path =",POSE_DIR)
print("Ply output path",OUT_PLY)
print("merged ply path",MERGED_PLY)

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
    print(f"[LOG] Loading pose file: {npz_path}")  # LOG
    data = np.load(npz_path)
    return data["rotation"], data["centre"]

def backproject_cube_face(img_path, depth_path, R_global, C_global, face_name):
    print(f"[LOG] Backprojecting face={face_name}")  # LOG
    print(f"[LOG] Image path={img_path}")           # LOG
    print(f"[LOG] Depth path={depth_path}")         # LOG

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    print(f"[LOG] Image shape: {img_np.shape}")     # LOG

    depth = np.load(depth_path)["depth"].astype(np.float32)
    print(f"[LOG] Depth shape: {depth.shape}")      # LOG

    if depth.shape != (H, W):
        print("[WARN] Depth resolution mismatch")   # LOG
        return np.zeros((0,3)), np.zeros((0,3))

    u = np.arange(0, W, STRIDE)
    v = np.arange(0, H, STRIDE)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()

    d = depth[vv, uu]
    valid = (d > MIN_DEPTH) & (d < MAX_DEPTH)
    print(f"[LOG] Valid depth pixels: {np.sum(valid)}")  # LOG

    if not np.any(valid):
        print("[LOG] No valid depth values")  # LOG
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

    print(f"[LOG] Points generated: {pts_world.shape[0]}")  # LOG

    return pts_world.astype(np.float32), cols.astype(np.float32)

def project_world_to_face(pts_world, R_global, C_global, face_name, H, W):
    R_face = get_face_rotation(face_name)
    pts_cam = (R_global @ (pts_world - C_global).T).T
    pts_face = (R_face.T @ pts_cam.T).T
    x, y, z = pts_face[:,0], pts_face[:,1], pts_face[:,2]
    valid = z > 0
    if not np.any(valid):
        print("[LOG] No points in front of camera for projection")  # LOG
        return np.array([]), np.array([])
    x, y, z = x[valid], y[valid], z[valid]
    u = (x / z * 0.5 + 0.5) * W
    v = (y / z * 0.5 + 0.5) * H
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print(f"[LOG] Projected points on face={face_name}: {np.sum(mask)}")  # LOG
    return u[mask].astype(np.int32), v[mask].astype(np.int32)

def main():
    print("\n[MV] Starting multiview fusion (cube faces)...")
    OUT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    pts_all, cols_all = [], []
    all_faces = sorted(CUBE_IMG_DIR.glob("*_depth_vis.png"))
    import re
    base_ids = sorted(set(
        re.match(r"(.+?)_\d+\.jpg_perspective_view_([a-z]+)_depth_vis\.png", p.name).group(1)
        for p in all_faces
        if re.match(r"(.+?)_\d+\.jpg_perspective_view_([a-z]+)_depth_vis\.png", p.name)
    ))

    print(f"[LOG] Found base_ids = {len(base_ids)}")   # LOG

    for base_id in base_ids:
        print(f"\n[LOG] Processing base_id={base_id}")  # LOG
        try:
            R_global, C_global = load_pose(base_id)
            print(f"[LOG] Loaded pose for {base_id}")   # LOG
        except Exception as e:
            print(f"[WARN] Missing pose for {base_id}: {e}")
            continue

        for face in FACES:
            print(f"[LOG] Checking face={face}")  # LOG

            img_path_candidates = list(CUBE_IMG_DIR.glob(f"{base_id}_*.jpg_perspective_view_{face}_depth_vis.png"))


            if not img_path_candidates:
                print(f"[WARN] No _depth_vis.png for {base_id} face={face}")
                continue
            img_path = img_path_candidates[0]
            depth_candidates = list(CUBE_DEPTH_DIR.glob(f"{base_id}_*.jpg_perspective_view_{face}_depth_meters.npz"))
            if not depth_candidates:
                print(f"[WARN] No depth_meters.npz for {base_id} face={face}")
                continue
            depth_path = depth_candidates[0]

            print(f"[LOG] Selected img={img_path.name}, depth={depth_path.name}")  # LOG

            pts, cols = backproject_cube_face(img_path, depth_path, R_global, C_global, face)
            if pts.shape[0] == 0:
                print(f"[LOG] No points generated for face {face}")  # LOG
                continue

            pts_all.append(pts)
            cols_all.append(cols)
            print(f"[LOG] Accumulated PTS: {sum(len(a) for a in pts_all)}")  # LOG

    if not pts_all:
        print("[ERR] No points produced.")
        return

    pts_all = np.vstack(pts_all)
    cols_all = np.vstack(cols_all)

    print(f"[LOG] Final fused cloud size: {pts_all.shape[0]} points")  # LOG

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_all)
    pc.colors = o3d.utility.Vector3dVector(cols_all)
    o3d.io.write_point_cloud(str(OUT_PLY), pc)
    print("[MV] Saved fused cloud:", OUT_PLY)

    # ======================================================
    # DEBUG overlays
    # ======================================================

    print("[MV] Creating debug projections...")
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
            img_candidates = list(CUBE_IMG_DIR.glob(
                f"{base_id}_*.jpg_perspective_view_{face}_depth_vis.png"
            ))
            if not img_candidates:
                continue
            
            img_path = img_candidates[0]

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to load image for debug overlay: {img_path}")  # LOG
                continue

            H, W = img.shape[:2]
            vis = img.copy()

            u, v = project_world_to_face(np.asarray(gen_cloud.points), R_global, C_global, face, H, W)
            print(f"[LOG] Debug proj (generated) pts={len(u)}")  # LOG
            vis[v, u] = (0, 0, 255)

            if merged_cloud is not None:
                um, vm = project_world_to_face(np.asarray(merged_cloud.points), R_global, C_global, face, H, W)
                print(f"[LOG] Debug proj (merged) pts={len(um)}")  # LOG
                vis[vm, um] = (0, 255, 0)

            out_path = OUT_DEBUG_DIR / f"{base_id}_{face}_overlay.png"
            cv2.imwrite(str(out_path), vis)
            print(f"[MV] Saved overlay projection: {out_path}")

    print("[MV] Debug projections complete.")

if __name__ == "__main__":
    main()
