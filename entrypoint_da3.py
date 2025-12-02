#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import time
from boto3.s3.transfer import TransferConfig
import boto3

USAGE = "Usage: python entrypoint_da3.py <user_id> <tour_id> <batch_size>"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def die(msg: str, code: int = 1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_symlink_for_da3(tour_id: str):
    src = Path("/mnt/shared/data") / tour_id
    dst_root = Path("/source/OpenSfM/data")
    dst = dst_root / tour_id
    try:
        dst_root.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve())
            print(f"[INFO] symlinked {dst} -> {src.resolve()}")
        else:
            print(f"[INFO] {dst} already exists (skipping symlink).")
    except PermissionError:
        print("[WARN] Could not create symlink under /source/OpenSfM.")


def upload_dir_to_s3(local_dir: Path, s3_prefix: str, bucket: str):
    """Upload only files corresponding to the current AWS_BATCH_JOB_ARRAY_INDEX."""
    s3 = boto3.client("s3")
    config = TransferConfig(use_threads=False)
    
    index = os.getenv("AWS_BATCH_JOB_ARRAY_INDEX")
    if index is None:
        print("[WARN] No AWS_BATCH_JOB_ARRAY_INDEX set — uploading all files.")
        index_pattern = None
    else:
        import re
        index_pattern = re.compile(rf"_{index}(?:_|\.|$)")
        print(f"[INFO] Upload filter enabled for index={index}")

    for path in local_dir.rglob("*"):
        if path.is_file():
            if index_pattern and not index_pattern.search(path.name):
                continue
            rel_path = path.relative_to(local_dir)
            key = f"{s3_prefix}/{rel_path}".replace("\\", "/")
            try:
                s3.upload_file(str(path), bucket, key, Config=config)
                print(f"[UPLOAD] {path} -> s3://{bucket}/{key}")
            except Exception as e:
                print(f"[WARN] Failed upload {path}: {e}")


def main():
    if len(sys.argv) < 4:
        die(USAGE)

    user_id, tour_id, batch_size = sys.argv[1], sys.argv[2], sys.argv[3]
    if not user_id.isdigit():
        die(f"Invalid user_id: {user_id}")
    if not tour_id.isdigit():
        die(f"Invalid tour_id: {tour_id}")
    if not batch_size.isdigit():
        die(f"Invalid batch_size: {batch_size}")

    print(f"[INFO] user_id={user_id}, tour_id={tour_id}, batch_size={batch_size}")

    s3_bucket = os.environ.get("S3_BUCKET")
    if not s3_bucket:
        die("S3_BUCKET env var is required")
    
    # Set and ensure environment directories
    ENV = {**os.environ, "DATA_ROOT": os.getenv("DATA_ROOT", "/mnt/shared/data")}
    ENV.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(ENV["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/opt/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/hf_cache")
    
    # Set default DA3 model (can be overridden via environment)
    os.environ.setdefault("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE")

    # Use shared EFS data
    DATA_ROOT = Path(ENV["DATA_ROOT"])
    undist_images = DATA_ROOT / tour_id / "images"
    print(f"[INFO] Using EFS-mounted data at {undist_images}")

    # AWS Batch array filtering
    # First check if USER provided override env var
    array_idx = int(os.getenv("FORCE_ARRAY_INDEX", os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", "0")))
    print(f"[INFO] AWS Batch array mode detected → index={array_idx}")

    # Find all images
    all_images = sorted([p for p in undist_images.glob("*") if p.suffix.lower() in IMG_EXTS])
    if not all_images:
        die(f"No images found under {undist_images}")

    # Filter only this index's 6 perspective views
    suffix = f"_{array_idx}.jpg"
    filtered_images = [p for p in all_images if p.name.endswith(suffix)]

    if not filtered_images:
        print(f"[WARN] No images match this array index ({array_idx}) under {undist_images}. Exiting gracefully.")
        sys.exit(0)

    print(f"[INFO] Found {len(filtered_images)} images for index={array_idx}")
    
    # Save filtered image list
    filtered_list_file = Path(f"/tmp/images_{array_idx}.txt")
    with open(filtered_list_file, "w") as f:
        for img in filtered_images:
            f.write(str(img) + "\n")

    # Add to environment for da3_batch.py
    ENV["DA3_IMAGE_LIST"] = str(filtered_list_file)

    ensure_symlink_for_da3(tour_id)

    dp_out_dir = DATA_ROOT / tour_id / "undistorted" / "undistort_depth_output"
    stitch_out_dir = DATA_ROOT / tour_id / "depth_output"
    dp_out_dir.mkdir(parents=True, exist_ok=True)
    stitch_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.time()
        print(f"(MAPERSIVE) Starting DA3 at {t0:.0f}")

        # ---- DA3 Inference ----
        subprocess.run(["python3", "da3_batch.py", "--data", tour_id, "--device", "cuda", "--batch", batch_size], 
                      check=True, env=ENV)
        time.sleep(2)

        # ---- Upload results ----

        print("[INFO] Uploading stitched outputs...")
        upload_dir_to_s3(stitch_out_dir, f"{user_id}/reconstruction/{tour_id}/depth_output", s3_bucket)

        t1 = time.time()
        print(f"(MAPERSIVE) Job finished at {t1:.0f}, duration {t1 - t0:.2f}s")

    except subprocess.CalledProcessError as e:
        die(f"Command {e.cmd} failed with exit {e.returncode}", code=e.returncode)


if __name__ == "__main__":
    main()
