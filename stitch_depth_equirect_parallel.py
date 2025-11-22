# tools/stitch_depth_equirect.py
import argparse
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble DepthPro cubemap face NPZ files into equirectangular depth maps, resized to original image size."
    )
    p.add_argument(
        "--data",
        help="Dataset ID (e.g., 60259). If provided, input dir defaults to /source/OpenSfM/data/<ID>/undistorted/undistort_depth_output.",
    )
    p.add_argument(
        "--in-dir",
        type=str,
        default=None,
        help="Directory with *_perspective_view_<face>_depth_meters.npz. Overrides --data if set.",
    )
    p.add_argument("--chunk-rows", type=int, default=512, help="Process rows per chunk to limit memory.")
    p.add_argument("--preview", action="store_true", help="Also write 16-bit PNG previews.")
    p.add_argument("--workers", type=int, default=0, help="Parallel workers (default: min(CPU, 8)).")
    return p.parse_args()

# -------------------- I/O helpers --------------------
def load_npz_depth(path: Path) -> np.ndarray:
    data = np.load(str(path))
    if isinstance(data, np.ndarray):
        arr = data
    else:
        for k in ("depth", "arr_0", "data", "map", "d"):
            if k in data:
                arr = data[k]
                break
        else:
            raise KeyError(f"No suitable array key found in {path}. Keys: {list(data.keys())}")
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Depth must be HxW. Got {arr.shape} in {path}")
    return arr.astype(np.float32, copy=False)

def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        off = (h - w) // 2
        return img[off : off + w, :]
    off = (w - h) // 2
    return img[:, off : off + h]

def prepare_faces_depth(faces_paths: Dict[str, Path]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load, center-crop to square, and resize all faces to a common F x F using NEAREST."""
    loaded = {}
    for k, p in faces_paths.items():
        if p is None:
            continue
        loaded[k] = load_npz_depth(p)

    if not loaded:
        raise RuntimeError("No faces provided to prepare_faces_depth")

    sizes = [min(a.shape[:2]) for a in loaded.values()]
    F = int(min(sizes))
    out = {}
    for k, a in loaded.items():
        sq = center_crop_square(a)
        out[k] = cv2.resize(sq, (F, F), interpolation=cv2.INTER_NEAREST)
    return out, F

# -------------------- Math (cube <-> equirect) --------------------
def direction_from_equirect_rows(x_start: int, x_end: int, y_start: int, y_end: int, out_w: int, out_h: int):
    xs = np.arange(x_start, x_end, dtype=np.float32)
    ys = np.arange(y_start, y_end, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    lon = (xg / out_w - 0.5) * (2.0 * np.pi)
    lat = (yg / out_h - 0.5) * np.pi
    X = np.sin(lon) * np.cos(lat)
    Y = np.sin(lat)
    Z = np.cos(lon) * np.cos(lat)
    return X, Y, Z

def face_uv_from_direction(X, Y, Z):
    ax, ay, az = np.abs(X), np.abs(Y), np.abs(Z)
    is_x = (ax >= ay) & (ax >= az)
    is_y = (ay > ax) & (ay >= az)
    is_z = (az > ax) & (az > ay)

    face_idx = np.full(X.shape, -1, dtype=np.int32)
    # right(+X)=0, left(-X)=1, top(+Y)=2, bottom(-Y)=3, front(+Z)=4, back(-Z)=5
    face_idx[(is_x) & (X > 0)] = 0
    face_idx[(is_x) & (X <= 0)] = 1
    face_idx[(is_y) & (Y > 0)] = 2
    face_idx[(is_y) & (Y <= 0)] = 3
    face_idx[(is_z) & (Z > 0)] = 4
    face_idx[(is_z) & (Z <= 0)] = 5

    eps = 1e-12
    u = np.zeros_like(X, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    m = (face_idx == 0); denom = ax[m] + eps; u[m] = (-Z[m] / denom) * 0.5; v[m] = (Y[m] / denom) * 0.5  # right
    m = (face_idx == 1); denom = ax[m] + eps; u[m] = (Z[m] / denom) * 0.5;  v[m] = (Y[m] / denom) * 0.5  # left
    m = (face_idx == 2); denom = ay[m] + eps; u[m] = (X[m] / denom) * 0.5;  v[m] = (-Z[m] / denom) * 0.5  # top
    m = (face_idx == 3); denom = ay[m] + eps; u[m] = (X[m] / denom) * 0.5;  v[m] = (Z[m] / denom) * 0.5   # bottom
    m = (face_idx == 4); denom = az[m] + eps; u[m] = (X[m] / denom) * 0.5;  v[m] = (Y[m] / denom) * 0.5   # front
    m = (face_idx == 5); denom = az[m] + eps; u[m] = (-X[m] / denom) * 0.5; v[m] = (Y[m] / denom) * 0.5   # back
    return face_idx, u, v

def cube_to_equirect_inverse_depth(
    faces_paths: Dict[str, Path], chunk_rows: int = 512
) -> Tuple[np.ndarray, int, int]:
    """
    faces_paths keys must be: {'front','back','left','right','top','bottom'}
    Values: Path to *_depth_meters.npz for each face.
    Output size is inferred from the face size: width=4F, height=2F, then caller may resize.
    """
    faces_sq, F = prepare_faces_depth(faces_paths)
    out_w, out_h = 4 * F, 2 * F
    out = np.zeros((out_h, out_w), dtype=np.float32)

    face_names = ["right", "left", "top", "bottom", "front", "back"]

    def uv_to_px(u, v):
        U = ((u + 0.5) * (F - 1)).astype(np.float32)
        V = ((v + 0.5) * (F - 1)).astype(np.float32)
        return U, V

    y = 0
    while y < out_h:
        y_end = min(y + chunk_rows, out_h)
        X, Y, Z = direction_from_equirect_rows(0, out_w, y, y_end, out_w, out_h)
        face_idx, u, v = face_uv_from_direction(X, Y, Z)
        U, V = uv_to_px(u, v)

        for idx, name in enumerate(face_names):
            src = faces_sq.get(name)
            if src is None:
                continue
            warped = cv2.remap(
                src, U, V, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            m = (face_idx == idx)
            out[y:y_end, :][m] = warped[m]

        y = y_end

    return out, out_w, out_h

# -------------------- Discovery & grouping --------------------
FACE_LABELS = {"front", "back", "left", "right", "top", "bottom"}
NPZ_PATTERN = re.compile(
    r"^(?P<stem>.+)\_perspective_view\_(?P<face>front|back|left|right|top|bottom)\_depth_meters\.npz$"
)

def find_groups(in_dir: Path) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for p in in_dir.glob("*_perspective_view_*_depth_meters.npz"):
        m = NPZ_PATTERN.match(p.name)
        if not m:
            continue
        stem = m.group("stem")  # includes original extension (e.g., .jpg)
        face = m.group("face")
        groups[stem][face] = p
    return groups

def ensure_face_map(face_map: Dict[str, Path]) -> Dict[str, Path]:
    missing: List[str] = [f for f in FACE_LABELS if f not in face_map]
    if missing:
        raise RuntimeError(f"Missing faces: {missing}")
    return face_map

def dataset_root_for(in_dir: Path) -> Path:
    """Find the dataset root so we can write to <root>/depth_output and read <root>/images."""
    cur: Optional[Path] = in_dir
    while cur and cur != cur.parent:
        if cur.name == "undistorted":
            return cur.parent
        cur = cur.parent
    return in_dir.parent

def original_image_size(ds_root: Path, stem_with_ext: str) -> Optional[Tuple[int, int]]:
    """Return (W,H) of the original pano in <root>/images/<stem_with_ext>, if it exists."""
    img_path = ds_root / "images" / stem_with_ext
    if img_path.is_file():
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    return None

# -------------------- Worker --------------------
def stitch_one(stem: str, face_map: Dict[str, Path], ds_root: Path, out_dir: Path,
               chunk_rows: int, preview: bool) -> Tuple[str, str, float]:
    """
    Returns (stem, output_npz_name, elapsed_seconds). Raises on failure.
    """
    t0 = time.perf_counter()

    faces = ensure_face_map(face_map)

    # NOTE: your face naming needed swaps (left<->right, top<->bottom)
    faces_paths = {
        "front": faces["front"],
        "back": faces["back"],
        "left": faces["right"],
        "right": faces["left"],
        "top": faces["bottom"],
        "bottom": faces["top"],
    }

    # 1) stitch at native cubemap resolution (4F x 2F)
    depth_equi, eq_w, eq_h = cube_to_equirect_inverse_depth(
        faces_paths, chunk_rows=chunk_rows
    )

    # 2) resize to ORIGINAL pano size
    target_wh = original_image_size(ds_root, Path(stem).name)
    if target_wh is not None:
        tw, th = target_wh
        if (eq_w, eq_h) != (tw, th):
            depth_equi = cv2.resize(
                depth_equi, (tw, th), interpolation=cv2.INTER_NEAREST
            ).astype(np.float32, copy=False)
            eq_w, eq_h = tw, th

    # 3) enforce depth range [0, 100] meters
    np.clip(depth_equi, 0.0, 100.0, out=depth_equi)

    # 4) Save WITHOUT the .jpg in the name → "<image_id>_<num>_depth.npz"
    stem_no_ext = Path(stem).stem
    base_name = f"{stem_no_ext}_depth.npz"
    npz_path = out_dir / base_name
    np.savez_compressed(npz_path, depth=depth_equi.astype(np.float32))

    # Optional preview (16-bit)
    if preview:
        d = depth_equi
        valid = (d > 0) & (d <= 100)
        if np.any(valid):
            preview_img = (np.clip(d, 0, 100) / 100.0 * 65535.0).astype(np.uint16)
            png_path = out_dir / f"{stem_no_ext}_depth_preview.png"
            cv2.imwrite(str(png_path), preview_img)

    elapsed = time.perf_counter() - t0
    return stem, npz_path.name, elapsed

# -------------------- Main --------------------
def main():
    args = parse_args()
    array_idx = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
    print(f"[stitch] AWS Batch array mode detected → index={array_idx}")

    # Prefer single-threaded OpenCV inside workers (outer pool controls parallelism)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Timers
    t_total_start = time.perf_counter()

    # Resolve input dir
    if args.in_dir:
        in_dir = Path(args.in_dir).resolve()
    elif args.data:
        data_root = os.getenv("DATA_ROOT", "/mnt/shared/data")
        in_dir = Path(data_root) / args.data / "undistorted" / "undistort_depth_output"
        in_dir = in_dir.resolve()
    else:
        raise SystemExit("Provide either --data <ID> or --in-dir <path>.")

    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    # Output to sibling folder 'depth_output' next to 'undistorted'
    ds_root = dataset_root_for(in_dir)
    out_dir = ds_root / "depth_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = find_groups(in_dir)
    # --- Filter groups by array index ---
    # Each group stem looks like "<uuid>_<index>.jpg"
    filtered_groups = {
        stem: faces for stem, faces in groups.items()
        if f"_{array_idx}.jpg" in stem
    }

    if not filtered_groups:
        print(f"[stitch][WARN] No groups found for array index={array_idx}. Exiting gracefully.")
        sys.exit(0)

    groups = filtered_groups
    print(f"[stitch] Found {len(groups)} group(s) for index={array_idx}")

    if not groups:
        raise SystemExit(f"No DepthPro face NPZ files found in: {in_dir}")
    
    print(f"[stitch] Filtering complete → proceeding with {len(groups)} group(s)")
    items = sorted(groups.items())
    total = len(items)
    print(f"[stitch] Found {total} groups under {in_dir}")
    print(f"[stitch] Output directory: {out_dir}")

    # Workers
    if args.workers and args.workers > 0:
        workers = args.workers
    else:
        workers = min(multiprocessing.cpu_count(), 8)
    print(f"[stitch] Using {workers} worker(s)")

    done = 0
    skipped = 0

    t_stitch_start = time.perf_counter()
    if workers == 1:
        # Sequential (baseline)
        for i, (stem, face_map) in enumerate(items, 1):
            try:
                stem_out, npz_name, t_el = stitch_one(stem, face_map, ds_root, out_dir, args.chunk_rows, args.preview)
                done += 1
                print(f"[{i}/{total}] {stem_out} → {npz_name} [TIME {t_el:.2f}s]")
            except Exception as ex:
                skipped += 1
                print(f"[stitch][WARN] Skipping '{stem}': {ex}")
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(stitch_one, stem, face_map, ds_root, out_dir, args.chunk_rows, args.preview)
                for stem, face_map in items
            ]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    stem_out, npz_name, t_el = f.result()
                    done += 1
                    print(f"[{i}/{total}] {stem_out} → {npz_name} [TIME {t_el:.2f}s]")
                except Exception as ex:
                    skipped += 1
                    print(f"[stitch][WARN] {ex}")

    t_stitch = time.perf_counter() - t_stitch_start
    t_total = time.perf_counter() - t_total_start

    ips = (done / t_stitch) if t_stitch > 0 else float("inf")
    print(f"[stitch][TIME] Stitch phase: {t_stitch:.2f}s for {done} group(s) ({ips:.2f} grp/s). Skipped: {skipped}")
    print(f"[stitch] Done. OK: {done}, Skipped: {skipped}, Out dir: {out_dir}")
    print(f"[stitch][TIME] Total wall time: {t_total:.2f}s")

if __name__ == "__main__":
    main()
