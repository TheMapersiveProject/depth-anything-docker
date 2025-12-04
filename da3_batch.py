# da3_batch.py
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

def parse_arguments():
    p = argparse.ArgumentParser(description="Run DA3 on undistorted images for an OpenSfM dataset.")
    p.add_argument("--data", required=True, help="Dataset name (e.g., 410704)")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"), 
                   help="Inference device (default: cuda; falls back to CPU if unavailable)")
    p.add_argument("--batch", type=int, default=1,
                   help="Batch size for inference (default: 1; if device=cpu this will be forced to 1)")
    return p.parse_args()

def get_face_rotation(face_name):
    """
    Returns the Rotation matrix (3x3) that rotates the global camera frame 
    to look at the specific cubemap face.
    Assumes the base camera is looking at +Z (Front).
    """
    # Standard Cubemap directions (View direction):
    # Front: +Z
    # Back:  -Z
    # Right: +X
    # Left:  -X
    # Top:   +Y
    # Bottom: -Y
    
    # But we need the rotation of the CAMERA axes (X-right, Y-down, Z-forward).
    # So we need a rotation R such that R * [0,0,1] = FaceDirection
    
    # Identity (Front)
    if face_name == 'front':
        return np.eye(3)
    
    # Rotate 180 around Y (Back)
    if face_name == 'back':
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        
    # Rotate -90 around Y (Right)
    if face_name == 'right':
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        
    # Rotate +90 around Y (Left)
    if face_name == 'left':
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        
    # Rotate -90 around X (Top)
    if face_name == 'top':
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        
    # Rotate +90 around X (Bottom)
    if face_name == 'bottom':
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    raise ValueError(f"Unknown face: {face_name}")


def _load_pose_matrices(batch_paths, rot_trans_dir):
    """
    Load rotation/translation for each image in the batch and construct extrinsics/intrinsics.
    """
    extrinsics_list = []
    intrinsics_list = []

    for path in batch_paths:
        # path format: uuid_index.jpg_perspective_view_face.jpg
        # npz format: shot_uuid_index.jpg.npz
        # We need to extract uuid_index from the path
        try:
            stem = path.stem # uuid_index.jpg_perspective_view_face
            # Split by .jpg_perspective_view_
            parts = stem.split(".jpg_perspective_view_")
            if len(parts) != 2:
                print(f"[DA3][WARN] Could not parse filename: {stem}, skipping pose")
                return None, None   
            image_id = parts[0] # uuid_index
            face_name = parts[1] # front, back, etc.
            npz_name = f"shot_{image_id}.jpg.npz"
            npz_path = rot_trans_dir / npz_name
            if not npz_path.exists():
                print(f"[DA3][WARN] Pose file not found: {npz_path}, skipping pose")
                return None, None
            data = np.load(npz_path)
            # R_global (3x3) and C_global (3,)
            R_global = data['rotation']
            C_global = data['centre']
            # Convert to World-to-Camera Translation: t = -R * C
            t_global = -R_global @ C_global
            # Get Face Rotation
            R_face_local = get_face_rotation(face_name)            
            # Combine Rotations
            # R_total = R_face_local @ R_global
            # t_total = R_face_local @ t_global
            
            R_total = R_face_local @ R_global
            t_total = R_face_local @ t_global
            
            # Build 4x4 Extrinsic Matrix
            E = np.eye(4)
            E[:3, :3] = R_total
            E[:3, 3] = t_total

            # DEBUG: Log first few poses to check coordinate system
            if len(extrinsics_list) == 0: 
                print(f"[DA3][DEBUG] Pose for {path.name}:")
                print(f"  R_global:\n{R_global}")
                print(f"  C_global (camera center): {C_global}")
                print(f"  t_global (W2C translation): {t_global}")
                print(f"  R_face ({face_name}):\n{R_face_local}")
                print(f"  R_total:\n{R_total}")
                print(f"  t_total:\n{t_total}")
                print(f"  Final Extrinsic (W2C):\n{E}")
                # Verify: Camera center should be at -R_total.T @ t_total
                C_verify = -R_total.T @ t_total
                print(f"  Verify camera center: {C_verify} (should match rotated C_global)")
                C_rotated = R_face_local @ C_global
                print(f"  Rotated C_global: {C_rotated}")
            
            # Build Intrinsics (90 deg FOV)
            # Assuming 512x512 or similar square images
            # We need to open the image to get W/H? Or assume based on config?
            # For now, let's load the image size inside the main loop or here?
            # To be safe, let's assume 1024 (or whatever process_res is) or just 1.0 normalized?
            # DA3 uses pixel coordinates. Let's use a dummy size and let DA3 resize?
            # NO, DA3 needs actual pixel focal length matching the input image.
            # We will peek at the image size.
            with Image.open(path) as img:
                W, H = img.size
                
            f = W / 2.0
            c = W / 2.0
            K = np.array([
                [f, 0, c],
                [0, f, c],
                [0, 0, 1]
            ])
            
            extrinsics_list.append(E)
            intrinsics_list.append(K)
            
        except Exception as e:
            print(f"[DA3][WARN] Error processing pose for {path.name}: {e}")
            return None, None

    return np.stack(extrinsics_list), np.stack(intrinsics_list)


# Global cache to prevent double loading
_model_cache = {}

def _has_camera_encoder(model) -> bool:
    """Check if model has camera encoder support (handles both regular and nested models)."""
    if not hasattr(model, 'model'):
        return False
    
    # Check if it's a nested model (has da3 branch)
    if hasattr(model.model, 'da3'):
        da3_branch = model.model.da3
        return (hasattr(da3_branch, 'cam_enc') and da3_branch.cam_enc is not None and
                hasattr(da3_branch, 'cam_dec') and da3_branch.cam_dec is not None)
    else:
        # Regular model
        return (hasattr(model.model, 'cam_enc') and model.model.cam_enc is not None and
                hasattr(model.model, 'cam_dec') and model.model.cam_dec is not None)


def _load_model(device: str):
    """Load DA3 model (configurable via DA3_MODEL env var)."""
    from depth_anything_3.api import DepthAnything3
    
    # Check cache first
    cache_key = (device, os.getenv("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE"))
    if cache_key in _model_cache:
        print(f"[DA3] Using cached model for {cache_key[1]}")
        return _model_cache[cache_key]
    
    model_name = os.getenv("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE")
    print(f"[DA3] Loading model: {model_name} (device={device}) ...")
    print(f"[DA3][DEBUG] DA3_MODEL env var: {repr(os.getenv('DA3_MODEL'))}")
    
    model = DepthAnything3.from_pretrained(model_name).to(device).eval()
    
    # Cache the model
    _model_cache[cache_key] = model
    
    # Detailed check for camera encoder
    print(f"[DA3][DEBUG] Model type: {type(model)}")
    print(f"[DA3][DEBUG] Model name attr: {getattr(model, 'model_name', 'N/A')}")
    print(f"[DA3][DEBUG] Has 'model' attr: {hasattr(model, 'model')}")
    
    if hasattr(model, 'model'):
        print(f"[DA3][DEBUG] model.model type: {type(model.model)}")
        model_type_name = type(model.model).__name__
        
        # Check if it's a nested model (has da3 branch)
        if hasattr(model.model, 'da3'):
            print(f"[DA3][DEBUG] Nested model detected - checking da3 branch")
            da3_branch = model.model.da3
            print(f"[DA3][DEBUG] da3 branch type: {type(da3_branch)}")
            print(f"[DA3][DEBUG] da3 has 'cam_enc' attr: {hasattr(da3_branch, 'cam_enc')}")
            if hasattr(da3_branch, 'cam_enc'):
                print(f"[DA3][DEBUG] da3.cam_enc value: {da3_branch.cam_enc}")
                print(f"[DA3][DEBUG] da3.cam_enc type: {type(da3_branch.cam_enc)}")
            if hasattr(da3_branch, 'cam_dec'):
                print(f"[DA3][DEBUG] da3.cam_dec value: {da3_branch.cam_dec}")
                print(f"[DA3][DEBUG] da3.cam_dec type: {type(da3_branch.cam_dec)}")
        else:
            # Regular model - check directly
            print(f"[DA3][DEBUG] Regular model - checking directly")
            print(f"[DA3][DEBUG] Has 'cam_enc' attr: {hasattr(model.model, 'cam_enc')}")
            if hasattr(model.model, 'cam_enc'):
                print(f"[DA3][DEBUG] cam_enc value: {model.model.cam_enc}")
                print(f"[DA3][DEBUG] cam_enc type: {type(model.model.cam_enc)}")
            if hasattr(model.model, 'cam_dec'):
                print(f"[DA3][DEBUG] cam_dec value: {model.model.cam_dec}")
                print(f"[DA3][DEBUG] cam_dec type: {type(model.model.cam_dec)}")
    
    # Check for camera encoder using helper function
    has_cam_enc = _has_camera_encoder(model)
    print(f"[DA3] Camera encoder available: {has_cam_enc}")
    
    if not has_cam_enc:
        print("[DA3][WARN] Camera encoder not available - multi-view pose conditioning disabled.")
        print("[DA3][WARN] Depth estimation will proceed without cross-view consistency.")
    
    return model


def _safe_inference(model, image_paths, extrinsics=None, intrinsics=None, process_res=1024):
    """
    Safely run model inference with fallback for alignment errors.
    
    For cubemap faces (rotation-only poses), the Umeyama alignment fails because
    there's no translation baseline. This function catches that error and retries
    without extrinsics as a fallback.
    """
    try:
        return model.inference(
            image=image_paths,
            process_res=process_res,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=False
        )
    except Exception as e:
        error_str = str(e)
        if "Degenerate covariance rank" in error_str or "Umeyama alignment" in error_str or "GeometryException" in error_str:
            # Alignment failed - retry without extrinsics
            # This loses camera conditioning but produces valid depth
            print(f"[DA3][WARN] Pose alignment failed, retrying without extrinsics.")
            return model.inference(
                image=image_paths,
                process_res=process_res,
                extrinsics=None,
                intrinsics=None
            )
        else:
            # Different error - re-raise
            raise


def _process_and_save_batch(model, device: str, batch_paths: list[Path], out_dir: Path, *, 
                            per_image_progress_start: int, total_images: int, rot_trans_dir: Path):
    """Process one batch and save outputs. Returns (completed_count, error_count, per_image_times)."""
    t_batch_start = time.perf_counter()
    completed = 0
    errors = 0
    per_image_times = []

    # Try to load poses
    extrinsics, intrinsics = _load_pose_matrices(batch_paths, rot_trans_dir)
    if extrinsics is not None:
        # Ensure model has camera encoder support
        if not _has_camera_encoder(model):
            print("[DA3][WARN] Model has no camera encoder; ignoring extrinsics/intrinsics.")
            extrinsics = None
            intrinsics = None
        else:
            print(f"[DA3][DEBUG] Extrinsics shape: {extrinsics.shape}, Intrinsics shape: {intrinsics.shape}")
    else:
        print("[DA3][WARN] Running without pose info (could not load matrices).")

    try:
        # Run inference with safe fallback for alignment errors
        with torch.no_grad():
            try:
                prediction = _safe_inference(
                    model,
                    image_paths=[str(p) for p in batch_paths],
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    process_res=1024
                )
            except Exception as align_error:
                # Check if it's the Umeyama alignment error
                error_str = str(align_error)
                if "Degenerate covariance rank" in error_str or "Umeyama alignment" in error_str or "GeometryException" in error_str:
                    print(f"[DA3][WARN] Pose alignment failed for batch (likely similar cubemap poses). Processing images individually.")
                    # Process each image individually with safe inference
                    predictions = []
                    for idx, path in enumerate(batch_paths):
                        single_ext = extrinsics[idx:idx+1] if extrinsics is not None else None
                        single_int = intrinsics[idx:idx+1] if intrinsics is not None else None
                        pred = _safe_inference(
                            model,
                            image_paths=[str(path)],
                            extrinsics=single_ext,
                            intrinsics=single_int,
                            process_res=1024
                        )
                        predictions.append(pred)
                    
                    # Combine predictions into a single batch-like result
                    # Create a mock prediction object with combined depths
                    from types import SimpleNamespace
                    combined_depths = np.stack([p.depth[0] for p in predictions])
                    prediction = SimpleNamespace()
                    prediction.depth = combined_depths
                else:
                    # Re-raise if it's a different error
                    raise
        
        # Save results for each image
        for idx, (path, depth) in enumerate(zip(batch_paths, prediction.depth), start=1):
            depth_abs = depth.astype(np.float32)
            stem = path.stem
            npz_path = out_dir / f"{stem}_depth_meters.npz"
            png_path = out_dir / f"{stem}_depth_vis.png"

            np.savez_compressed(npz_path, depth=depth_abs)

            # Visualization
            d = depth_abs
            d_norm = (d - d.min()) / max(1e-8, (d.max() - d.min()))
            Image.fromarray((d_norm * 255).astype("uint8")).save(png_path)

            done_idx = per_image_progress_start + idx - 1
            print(f"[{done_idx}/{total_images}] {path.name} → {npz_path.name}, {png_path.name}")
            completed += 1

        t_batch = time.perf_counter() - t_batch_start
        if completed > 0:
            per_image_times.extend([t_batch / completed] * completed)

    except RuntimeError as re:
        # OOM fallback: try one-by-one
        if "out of memory" in str(re).lower() and len(batch_paths) > 1 and device == "cuda":
            print(f"[DA3][WARN] OOM on batch starting {batch_paths[0].name}. Falling back to single-image processing.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            for idx, path in enumerate(batch_paths, start=1):
                try:
                    t_img_start = time.perf_counter()
                    # For single image, extract corresponding extrinsics/intrinsics if available
                    single_extrinsics = None
                    single_intrinsics = None
                    if extrinsics is not None and intrinsics is not None:
                        single_extrinsics = extrinsics[idx-1:idx]  # Keep batch dimension
                        single_intrinsics = intrinsics[idx-1:idx]
                    
                    # Use safe inference with automatic fallback
                    prediction = _safe_inference(
                        model,
                        image_paths=[str(path)],
                        extrinsics=single_extrinsics,
                        intrinsics=single_intrinsics,
                        process_res=1024
                    )
                    depth_abs = prediction.depth[0].astype(np.float32)

                    stem = path.stem
                    npz_path = out_dir / f"{stem}_depth_meters.npz"
                    png_path = out_dir / f"{stem}_depth_vis.png"
                    np.savez_compressed(npz_path, depth=depth_abs)

                    d = depth_abs
                    d_norm = (d - d.min()) / max(1e-8, (d.max() - d.min()))
                    Image.fromarray((d_norm * 255).astype("uint8")).save(png_path)

                    done_idx = per_image_progress_start + idx - 1
                    t_img = time.perf_counter() - t_img_start
                    print(f"[{done_idx}/{total_images}] {path.name} → {npz_path.name}, {png_path.name} [TIME {t_img:.2f}s]")
                    per_image_times.append(t_img)
                    completed += 1

                except Exception as e1:
                    errors += 1
                    print(f"[DA3][ERROR] {path.name}: {e1}")
        else:
            errors += len(batch_paths)
            print(f"[DA3][ERROR] Batch starting {batch_paths[0].name}: {re}")
    except Exception as e:
        import traceback
        errors += len(batch_paths)
        print(f"[DA3][ERROR] Batch starting {batch_paths[0].name}: {e}")
        print("[DA3][TRACE]", traceback.format_exc())

    return completed, errors, per_image_times


def main():
    print("[DA3] VERSION: v4 (Safe Inference + Pose Debug)")
    args = parse_arguments()
    t_total_start = time.perf_counter()

    data_root = os.getenv("DATA_ROOT", "/mnt/shared/data")
    data_path = os.path.join(data_root, args.data)
    undistort_path = os.path.join(data_path, "undistorted")
    rot_trans_dir = Path(data_path) / "rot_trans_matrix_npy"
    images_dir = Path(undistort_path) / "images"
    out_dir = Path(undistort_path) / "undistort_depth_output"
    print(f"[DA3] Using data root: {data_root}")
    print(f"[DA3] Images dir: {data_path}/undistorted/images")
    print(f"[DA3] Poses dir: {rot_trans_dir}")

    if not images_dir.is_dir():
        t_total = time.perf_counter() - t_total_start
        raise FileNotFoundError(f"Images folder not found: {images_dir} (TIME total so far {t_total:.2f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("[DA3][WARN] CUDA was requested but not available. Falling back to CPU.")
    print(f"[DA3] Using device: {device}")

    # Batch size
    batch_size = max(1, int(args.batch))
    if device == "cpu":
        batch_size = 1
    print(f"[DA3] Batch size: {batch_size}")

    # Load model
    print(f"[DA3] Loading model: DA3METRIC-LARGE ...")
    t_load_start = time.perf_counter()
    model = _load_model(device)
    t_load = time.perf_counter() - t_load_start
    print(f"[DA3][TIME] Model load time: {t_load:.2f}s")

    # Collect images
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
    list_file = os.getenv("DA3_IMAGE_LIST")
    if list_file and os.path.isfile(list_file):
        with open(list_file, "r") as f:
            images = [Path(line.strip()) for line in f if line.strip()]
        print(f"[DA3] Using filtered image list ({len(images)} images)")

    print(f"[DA3] Found {len(images)} image(s) in {images_dir}")

    if not images:
        t_total = time.perf_counter() - t_total_start
        print(f"[DA3] Nothing to do. [TIME total {t_total:.2f}s]")
        return

    # Inference loop with batching
    t_inf_start = time.perf_counter()
    per_image_times = []
    completed = 0
    errors = 0
    total = len(images)
    next_progress_idx = 1

    for start in range(0, total, batch_size):
        batch_paths = images[start:start + batch_size]
        c, e, times = _process_and_save_batch(
            model, device, batch_paths, out_dir,
            per_image_progress_start=next_progress_idx,
            total_images=total,
            rot_trans_dir=rot_trans_dir
        )
        completed += c
        errors += e
        per_image_times.extend(times)
        next_progress_idx += len(batch_paths)

    # Timing summary
    t_inf = time.perf_counter() - t_inf_start
    t_total = time.perf_counter() - t_total_start

    if completed > 0:
        est_avg = t_inf / completed if completed else 0.0
        ips = completed / t_inf if t_inf > 0 else float("inf")
        print(f"[DA3][TIME] Inference loop: {t_inf:.2f}s for {completed} image(s) (~{est_avg:.2f}s/img, {ips:.2f} img/s)")
    else:
        print(f"[DA3][TIME] Inference loop: {t_inf:.2f}s (no successful images)")

    print(f"[DA3] Done. OK: {completed}, ERR: {errors}")
    print(f"[DA3][TIME] Total wall time: {t_total:.2f}s (includes load, I/O, etc.)")


if __name__ == "__main__":
    main()