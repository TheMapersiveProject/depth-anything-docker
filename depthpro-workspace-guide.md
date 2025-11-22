DepthPro Docker Workspace Guide
================================

**Repository:** `depthpro_docker_image` (GitHub)

This document explains the DepthPro depth estimation workflow implemented in the `depthpro_docker_image` GitHub repository. It covers what the scripts in that workspace do, how they work together, data flow conventions, and technical details observable in that codebase.

**Relationship to SfM Pipeline:** The DepthPro job described here integrates with the broader OpenSfM reconstruction pipeline infrastructure (documented separately in `sfm-batch-job-overview.md` and `sfm-stage-efs-io.md`). The DepthPro job consumes outputs from the SfM pipeline's Stage 4 (Undistort & Depthmaps) and produces depth estimation artifacts that are stored in parallel to the SfM reconstruction outputs.

Overview
--------

The `depthpro_docker_image` repository provides a containerized AWS Batch job that:

1. Reads undistorted cubemap perspective images from EFS (6 faces per panorama)
2. Runs Apple DepthPro ML inference to generate metric depth maps for each face
3. Stitches the 6 depth faces back into equirectangular depth maps
4. Uploads results to S3

The workflow is designed for parallel execution via AWS Batch array jobs, where each array index processes one panorama's worth of images.

Container Entry Point
---------------------

**Main script:** `entrypoint.py`

**Usage:**
```bash
python entrypoint.py <user_id> <tour_id> <batch_size>
```

**Required environment:**
- `S3_BUCKET` (required): target S3 bucket for uploads
- `AWS_BATCH_JOB_ARRAY_INDEX` (optional): array job index, defaults to 0
- `DATA_ROOT` (optional): EFS mount point, defaults to `/mnt/shared/data`
- `HF_HOME` / `TRANSFORMERS_CACHE` (optional): defaults to `/opt/hf_cache`

**What it does:**

1. **Validates inputs:** user_id, tour_id, batch_size must be numeric
2. **Locates data:** Expects undistorted images at `DATA_ROOT/<tour_id>/undistorted/images/`
3. **Filters by array index:** Only processes images matching `_<index>.jpg_perspective_view_*.jpg`
4. **Creates symlink:** Links `/source/OpenSfM/data/<tour_id>` → `/mnt/shared/data/<tour_id>` (legacy path support)
5. **Runs DepthPro inference:** Calls `depthpro_batch.py` with filtered image list
6. **Stitches cubemap to equirect:** Calls `stitch_depth_equirect_parallel.py`
7. **Uploads results to S3:** Uploads only files matching current array index pattern

Input Data Conventions
----------------------

**Expected input structure:**
```
DATA_ROOT/
└── <tour_id>/
    ├── images/                          # Original panoramas (used for size reference)
    │   └── <uuid>_<index>.jpg
    └── undistorted/
        └── images/                      # Cubemap perspective faces (INPUT)
            ├── <uuid>_<index>.jpg_perspective_view_front.jpg
            ├── <uuid>_<index>.jpg_perspective_view_back.jpg
            ├── <uuid>_<index>.jpg_perspective_view_left.jpg
            ├── <uuid>_<index>.jpg_perspective_view_right.jpg
            ├── <uuid>_<index>.jpg_perspective_view_top.jpg
            └── <uuid>_<index>.jpg_perspective_view_bottom.jpg
```

**File naming pattern:**
- Original pano stem: `<uuid>_<index>.jpg`
- Cubemap face format: `<stem>_perspective_view_<face>.jpg`
- The `<index>` field maps to `AWS_BATCH_JOB_ARRAY_INDEX` for parallelization

**Face labels:** `front`, `back`, `left`, `right`, `top`, `bottom`

Array Job Filtering
-------------------

Both `entrypoint.py` and `stitch_depth_equirect_parallel.py` filter work by array index:

**Image filtering (entrypoint.py:118-119):**
```python
suffix = f"_{array_idx}.jpg_perspective_view_"
filtered_images = [p for p in all_images if suffix in p.name]
```

**Upload filtering (entrypoint.py:49-62):**
- Uses regex `_{index}(?:_|\.|$)` to only upload files belonging to this index
- Prevents race conditions when multiple array jobs write to shared output directories

**Stitch filtering (stitch_depth_equirect_parallel.py:297-299):**
```python
filtered_groups = {
    stem: faces for stem, faces in groups.items()
    if f"_{array_idx}.jpg" in stem
}
```

If no images match the current index, scripts exit gracefully with code 0.

DepthPro Inference Stage
-------------------------

**Script:** `depthpro_batch.py`

**Usage:**
```bash
python depthpro_batch.py --data <tour_id> [--device cuda|cpu] [--batch N]
```

**Model:** `apple/DepthPro-hf` from HuggingFace Transformers
- Pre-downloaded in Dockerfile build (lines 42-44)
- Requires transformers==4.49.0
- Uses `DepthProImageProcessorFast` and `DepthProForDepthEstimation`

**Input:** Reads from `DATA_ROOT/<tour_id>/undistorted/images/`
- Honors `DEPTHPRO_IMAGE_LIST` env var (set by entrypoint.py) to process only filtered images
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

**Output directory:** `DATA_ROOT/<tour_id>/undistorted/undistort_depth_output/`

**Output files per image:**
- `<stem>_depth_meters.npz` - compressed NumPy array with float32 depth in meters (key: "depth")
- `<stem>_depth_vis.png` - normalized 8-bit visualization (for debugging)

**Batch processing:**
- Default batch size: 1
- CPU mode forces batch_size=1
- CUDA mode supports larger batches with automatic mixed precision (AMP)
- OOM fallback: if batch fails with "out of memory", retries images one-by-one

**Performance:**
- Reports per-image timing and throughput (img/s)
- Times are logged with `[TIME]` prefix for easy parsing

Cubemap Stitching Stage
------------------------

**Script:** `stitch_depth_equirect_parallel.py`

**Usage:**
```bash
python stitch_depth_equirect_parallel.py --data <tour_id> [--preview] [--workers N] [--chunk-rows N]
```

**Input:** `DATA_ROOT/<tour_id>/undistorted/undistort_depth_output/*_perspective_view_*_depth_meters.npz`

**Output directory:** `DATA_ROOT/<tour_id>/depth_output/`

**Output files per panorama:**
- `<uuid>_<index>_depth.npz` - equirectangular depth map, resized to original pano dimensions
- `<uuid>_<index>_depth_preview.png` - optional 16-bit PNG visualization (if --preview)

**Process:**

1. **Face grouping:** Groups NPZ files by stem using regex:
   ```python
   r"^(?P<stem>.+)_perspective_view_(?P<face>front|back|left|right|top|bottom)_depth_meters\.npz$"
   ```

2. **Face preparation:**
   - Loads all 6 faces
   - Center-crops each to square
   - Resizes to common size F×F using INTER_NEAREST (preserves depth discontinuities)

3. **Coordinate remapping (CRITICAL):**
   - Swaps left ↔ right faces (lines 219, 220)
   - Swaps top ↔ bottom faces (lines 221, 222)
   - Maps equirectangular (lon, lat) → cubemap (face_idx, u, v)
   - Uses `cv2.remap` with INTER_NEAREST interpolation

4. **Resizing to original:**
   - Looks up original image size from `DATA_ROOT/<tour_id>/images/<stem>`
   - Resizes equirect depth map to match using INTER_NEAREST

5. **Depth clamping:**
   - Clips all depth values to [0.0, 100.0] meters (line 241)

6. **Output naming:**
   - Strips `.jpg` extension from stem before saving
   - Output: `<uuid>_<index>_depth.npz` (NOT `<uuid>_<index>.jpg_depth.npz`)

**Parallelization:**
- Default workers: min(CPU_count, 8)
- Forces OpenCV to single-threaded mode (cv2.setNumThreads(1)) to avoid thread contention
- Uses ThreadPoolExecutor for parallel stitching

**Memory optimization:**
- Processes equirect rows in chunks (default: 512 rows at a time)
- Prevents loading entire 4F×2F output into memory at once

Face Coordinate System
----------------------

**Observed face swap in stitch_depth_equirect_parallel.py (lines 216-222):**

```python
faces_paths = {
    "front": faces["front"],
    "back": faces["back"],
    "left": faces["right"],    # SWAPPED
    "right": faces["left"],     # SWAPPED
    "top": faces["bottom"],     # SWAPPED
    "bottom": faces["top"],     # SWAPPED
}
```

This indicates the input face labels (from perspective view file names) use a different coordinate convention than the cubemap-to-equirect mapping math. The swap ensures correct spatial orientation in the output equirectangular depth map.

**Cubemap face indexing (internal, lines 103-109):**
- 0 = right (+X)
- 1 = left (-X)
- 2 = top (+Y)
- 3 = bottom (-Y)
- 4 = front (+Z)
- 5 = back (-Z)

**Equirectangular projection:**
- Width = 4F, Height = 2F (at native cubemap resolution)
- Longitude: [-π, π] (left to right)
- Latitude: [-π/2, π/2] (top to bottom)

S3 Upload & Output Locations
-----------------------------

**Upload targets (entrypoint.py:157-168):**

1. **Raw DepthPro outputs:**
   - Local: `DATA_ROOT/<tour_id>/undistorted/undistort_depth_output/`
   - S3: `s3://$S3_BUCKET/<user_id>/reconstruction/<tour_id>/undistorted/undistort_depth_output/`

2. **Stitched equirect outputs:**
   - Local: `DATA_ROOT/<tour_id>/depth_output/`
   - S3: `s3://$S3_BUCKET/<user_id>/reconstruction/<tour_id>/depth_output/`

**Upload behavior:**
- Uses `boto3.s3.transfer.TransferConfig(use_threads=False)` for single-threaded uploads
- Only uploads files matching current `AWS_BATCH_JOB_ARRAY_INDEX` (via regex filter)
- Preserves directory structure under S3 prefix

**Helper scripts (NOT called by entrypoint.py):**
- `upload_depthpro_zip.py` - zips and uploads results (alternative to direct upload)
- `unzip_depthpro_in_s3.py` - expands zips in S3, then deletes zip

Docker Image Details
--------------------

**Base image:** `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`

**Key dependencies (requirements.txt + Dockerfile):**
- `torch` (from base image: 2.3.1)
- `transformers==4.49.0` (pinned for DepthPro compatibility)
- `numpy==1.26.4` (CRITICAL: must be <2.0 for torch 2.x compatibility)
- `timm==1.0.9` (vision models)
- `opencv-python-headless>=4.8`
- `Pillow>=9.5`
- `boto3==1.21.4`
- `accelerate>=0.34,<0.36`
- `hf-transfer>=0.1.6` (fast HF downloads)

**Directory structure:**
```
/app/                           # Application scripts
/source/OpenSfM/data/           # Symlink target (legacy path)
/data/ → /app/data              # Workspace (symlinked to /app/data)
/opt/hf_cache/                  # HuggingFace model cache
/mnt/shared/data/               # EFS mount point (mounted at runtime)
```

**User:** `runner` (non-root, UID/GID match EFS access point)

**Volumes:**
- `/opt/hf_cache` - model cache persistence
- `/data` - workspace (typically bind-mounted to EFS)

**Entrypoint:** `/usr/bin/tini -- python3 /app/entrypoint.py`

Environment Variables Reference
-------------------------------

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `S3_BUCKET` | Yes | - | Target S3 bucket for uploads |
| `DATA_ROOT` | No | `/mnt/shared/data` | EFS workspace root |
| `AWS_BATCH_JOB_ARRAY_INDEX` | No | `0` | Array job index for parallelization |
| `HF_HOME` | No | `/opt/hf_cache` | HuggingFace cache directory |
| `TRANSFORMERS_CACHE` | No | `/opt/hf_cache` | Transformers model cache |
| `DEPTHPRO_IMAGE_LIST` | No | - | Path to file with filtered image paths |
| `MPLCONFIGDIR` | No | `/tmp/mpl` | Matplotlib config (set by entrypoint) |
| `TOKENIZERS_PARALLELISM` | No | `false` | Disable tokenizer warnings |
| `HF_HUB_ENABLE_HF_TRANSFER` | No | `1` | Use fast hf-transfer for downloads |

Error Handling & Edge Cases
----------------------------

**OOM recovery (depthpro_batch.py:116-154):**
- Catches `RuntimeError` with "out of memory" message
- Falls back to single-image processing
- Calls `torch.cuda.empty_cache()` before retry
- Reports per-image timing even in fallback mode

**Missing images:**
- If no images match array index, exits with code 0 (success, not failure)
- Logs `[WARN] No images match this array index`

**Missing faces:**
- Stitching requires all 6 faces present
- Raises `RuntimeError` if any face is missing (lines 182-183)

**Invalid arguments:**
- entrypoint.py validates user_id, tour_id, batch_size are numeric
- Dies with code 1 if validation fails

**Missing S3_BUCKET:**
- Dies immediately with error message

**Subprocess failures:**
- entrypoint.py catches `CalledProcessError`
- Exits with subprocess's return code
- Does NOT upload partial results if DepthPro or stitch fails

Test Coverage
-------------

**Test files:**
- `tests/test_depthpro_batch.py` - unit tests for DepthPro inference
- `tests/test_entrypoint.py` - integration tests for main workflow

**Testing approach:**
- Mocks heavy dependencies (torch, transformers, boto3)
- Tests argument validation, error handling, subprocess calls
- Verifies file creation and upload logic
- Does NOT test actual ML inference or S3 operations

**Running tests:**
```bash
python -m pytest tests/
```

Tribal Knowledge & Gotchas
---------------------------

1. **The symlink hack (entrypoint.py:26-38):**
   - Creates `/source/OpenSfM/data/<tour_id>` → `/mnt/shared/data/<tour_id>`
   - Purpose unclear from code alone, likely legacy path requirement
   - PermissionError is caught and logged but doesn't fail the job

2. **Face swap is intentional:**
   - The left↔right, top↔bottom swap in stitching is NOT a bug
   - Required for correct spatial orientation in output
   - Changing this will flip the depth maps

3. **Depth clamping to [0, 100m]:**
   - Hardcoded in stitch_depth_equirect_parallel.py:241
   - No configuration option
   - Values outside this range are clipped

4. **Output filename strips .jpg:**
   - Input: `abc_0.jpg_perspective_view_front.jpg`
   - Output: `abc_0_depth.npz` (NOT `abc_0.jpg_depth.npz`)
   - Done via `Path(stem).stem` (line 244)

5. **INTER_NEAREST everywhere:**
   - All depth resampling uses nearest-neighbor interpolation
   - Bilinear/bicubic would blur depth discontinuities (edges, occlusions)
   - This is intentional for preserving sharp depth transitions

6. **NumPy version constraint:**
   - MUST be <2.0 for compatibility with torch 2.1.x–2.3.x
   - Breaking this will cause runtime errors

7. **aws_download.py exists but isn't used:**
   - Script has logic to download undistorted images from S3
   - entrypoint.py does NOT call it (assumes data already on EFS)
   - Possibly for alternative execution modes or legacy compatibility

8. **Upload is index-filtered:**
   - Even though all array jobs write to shared EFS output directories
   - Each job only uploads its own files to S3 (via regex filtering)
   - Prevents race conditions and duplicate uploads

Performance Notes
-----------------

**Typical timing (observed from TIME logs):**
- Model load: ~2-5 seconds (cached) to ~30s (first download)
- DepthPro inference: ~0.5-2s per image (GPU), ~5-10s (CPU)
- Stitching: ~1-3s per panorama (6 faces)
- Upload: depends on file size and network

**Memory usage:**
- DepthPro batch processing: ~4-8 GB GPU memory (batch=1)
- Stitching: ~2-4 GB RAM (with 512-row chunks)
- Peak occurs during face loading before stitching

**Parallelization:**
- Array jobs enable horizontal scaling (one job per panorama)
- Stitching uses ThreadPoolExecutor (default: 8 workers)
- No inter-job communication required

Quick Reference Commands
------------------------

**Build Docker image:**
```bash
docker build -t depthpro:latest .
```

**Run locally (mock):**
```bash
export S3_BUCKET=test-bucket
export DATA_ROOT=/tmp/data
export AWS_BATCH_JOB_ARRAY_INDEX=0

# Prepare test data structure
mkdir -p /tmp/data/410704/undistorted/images
mkdir -p /tmp/data/410704/images

# Run
python entrypoint.py 10477 410704 2
```

**Test individual components:**
```bash
# DepthPro only
python depthpro_batch.py --data 410704 --device cpu --batch 1

# Stitch only
python stitch_depth_equirect_parallel.py --data 410704 --preview --workers 4
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

**Check output structure:**
```bash
tree /mnt/shared/data/410704/undistorted/undistort_depth_output/
tree /mnt/shared/data/410704/depth_output/
```

