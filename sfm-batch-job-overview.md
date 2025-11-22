AWS Batch SFM Pipeline Overview
================================

**System:** AWS Batch-based Structure-from-Motion (SfM) reconstruction pipeline infrastructure

This document describes how the Structure-from-Motion (SfM) Batch workflow is organized across the SfM pipeline infrastructure. Use it as a quick reference when discussing job orchestration, data flow, infrastructure layout, and environment requirements with other teams. It covers what each Batch stage does, the data it exchanges, how environment variables control behaviour, and the AWS scaffolding (queues, compute environments, IAM, storage, observability) that keeps the pipeline running.

**Note:** This document describes infrastructure and job definitions that are separate from any specific application repository. The stages referenced here (Stage 1–5) are part of the OpenSfM reconstruction pipeline infrastructure.

Infrastructure Overview
-----------------------

- **Terraform layout:** All SfM infrastructure is provisioned via modules under `modules/` and instantiated per environment in `environments/<env>/`. Remote state exports VPC, subnet, and shared security group details so Batch resources land in the same network perimeter.
- **Job queues & definitions:** Each stage has a dedicated job definition (`<stage>-definition_<env>`) and queue (`<stage>-queue_<env>`). Queues point at one or more compute environment ARNs in priority order, enabling failover between capacity pools. Queue priority defaults to `1`; orchestration is handled by whichever system submits jobs (outside the infrastructure modules), so prioritisation is driven by submission order and dependency rules.
- **ECR repositories:** Container images live in ECR repos named `<stage>_<env>` (for example `opensfm-stage3_reconstruction_prod`). Repositories are encrypted (AES-256 by default), `scan_on_push` is disabled unless explicitly overridden, and optional lifecycle policies control image retention. CI/CD pipelines push immutable tags that job definitions consume.
- **EFS workspace:** All stages share the `mapersive-batch-efs-<env>` file system (access point `/batch-shared`) mounted at `/mnt/shared`. Terraform manages lifecycle rules (1-day IA, 7-day archive) and automated backups. A dedicated security group (`efs-nfs-sg`) grants NFS access only from the Batch worker security group.
- **Identity & permissions:** Each job definition gets its own execution role (`<stage>-<env>-execution`) and job role (`<stage>-<env>`). Base policies include `AmazonECSTaskExecutionRolePolicy` plus broad S3, SQS, and CloudWatch access. When EFS is enabled, least-privilege policies grant `elasticfilesystem:Client*` on the managed access point. Containers run as UID/GID `1000` to match the access point POSIX identity unless overridden.
- **Compute environments:** Four capacity pools back the SfM workloads:
  - `light-compute-env`: c6id.4xlarge instances (0–256 vCPUs) for low/medium stages.
  - `heavy-compute-env`: mix of c5.{4x,9x,12x,18x,24x}large (0–4096 vCPUs) for reconstruction-heavy stages.
  - `custom-ami-50gb-ebs-heavy`: c5.{2x,4x,9x}large instances booted from an AMI with pre-installed dependencies (0–1024 vCPUs).
  - `managed-gpu-enabled-compute-env`: g4dn.* family (0–256 vCPUs) for GPU-enabled jobs such as DepthPro.
  Each environment shares a launch template with a 500 GB gp2 root volume and host `/tmp` bind-mounted into containers when requested.
- **Networking:** Batch compute environments attach to private subnets exported by the shared network stack and reuse the `batch_sg_id` security group. Outbound internet uses NAT; inbound is blocked except for NFS traffic to EFS via the security group rule above.
- **Observability:** Job roles have CloudWatch Logs permissions; container images must configure log drivers (awslogs) to emit stdout/stderr. CloudWatch metrics (queue length, runnable jobs, vCPU usage) are enabled by default for Batch; alarms live under `modules/cloudwatch/*` if you need to wire notifications. Tagging across resources (`Managed=Terraform`, `Service=batch`) keeps Cost Explorer aggregation consistent.

Stage-to-Infra Mapping
----------------------

| Stage | Job definition | Queue | Compute environments (priority order) | vCPU / Memory | Timeout |
|-------|----------------|-------|---------------------------------------|---------------|---------|
| Stage 1 – Preprocess & Feature Extraction | `opensfm_stage1_preprocess-definition_<env>` | `opensfm_stage1_preprocess-queue_<env>` | [`light-compute-env`] | 10 vCPU / 28 GB | 14,400 s |
| Stage 2 – Submodel Reconstruction | `opensfm_stage2_submodels-definition_<env>` | `opensfm_stage2_submodels-queue_<env>` | [`heavy-compute-env`] | 72 vCPU / 140 GB | 28,800 s |
| Stage 3 – Full Reconstruction | `opensfm_stage3_reconstruction-definition_<env>` | `opensfm_stage3_reconstruction-queue_<env>` | [`heavy-compute-env`] | 72 vCPU / 140 GB | 28,800 s |
| Stage 4 – Undistort & Depthmaps | `opensfm_stage4_undistort_depth-definition_<env>` | `opensfm_stage4_undistort_depth-queue_<env>` | [`heavy-compute-env`] | 72 vCPU / 140 GB | 28,800 s |
| Stage 5 – Statistics & Reporting | `opensfm_stage5_stats_report-definition_<env>` | `opensfm_stage5_stats_report-queue_<env>` | [`light-compute-env`] | 8 vCPU / 16 GB | 14,400 s |

Other Batch workloads in the infrastructure (segmentation, make\_tiles, DepthPro, etc.) reuse the same compute environments, tagging scheme, and EFS mount conventions.

All stage job definitions mount EFS (`enable_efs=true`) with IAM authorization unless noted. Host `/tmp` can be mounted by toggling `enable_host_tmp_mount` when extra scratch storage is needed.

IAM, Security & Integrations
----------------------------

- **Job roles:** Every stage’s job role inherits `AmazonS3FullAccess`, `AmazonSQSFullAccess`, and `CloudWatchFullAccess`. The broad policies simplify experimentation but can be scoped down (e.g., to the SfM prefixes or specific queues) if least-privilege hardening is required.
- **Execution roles:** Execution roles load container images from ECR via the managed `AmazonECSTaskExecutionRolePolicy`. When `enable_efs=true`, Terraform adds inline policies that allow `elasticfilesystem:ClientMount/Write/RootAccess` for the specific file system and access point.
- **EFS access:** Access points enforce POSIX UID/GID `1000` (plus optional secondary groups) and permissions `0750`. Keep container users aligned; set `run_as_uid`/`run_as_gid` to `null` only when the image entrypoint manages ownership itself.
- **S3 buckets:** `S3_BUCKET` varies by environment (`mapersive-image-bucket-dev|staging`, `tour-upload-bucket` in prod). Bucket policies live in the shared S3 infrastructure module and allow the Batch roles to `GetObject`, `PutObject`, and `ListBucket` on the required prefixes.
- **SQS integrations:** Stage 5 publishes remediation jobs via `SQS_QUEUE_URL` (missing panoramas) and `TOUR_GPS_ERROR_HANDLER_SQS_QUEUE_URL` (GPS fixes). Queue URLs are environment-scoped and require the `AmazonSQSFullAccess` policy above; if you tighten permissions, ensure these specific ARNs stay whitelisted.
- **Privileged mode:** Only the dev Stage 1 job currently runs with `privileged=true` to prototype device-bound workflows. Production/staging stay unprivileged; review Dockerfiles before enabling privileged mode elsewhere.
- **Secrets & parameters:** All stage configuration is supplied via static environment variables in Terraform. If you introduce credentials, prefer AWS Secrets Manager or SSM Parameter Store and inject via an entrypoint script rather than hardcoding values in Terraform state.

Environment-Specific Defaults
-----------------------------

- **Regions:** All environments run in `eu-west-3`; batch jobs set `AWS_DEFAULT_REGION` explicitly where SDK calls are made (Stage 5 today).
- **Container images:** ECR tags follow `<repo>_<env>:latest`. Promotion flows should push a new SHA-tagged image, update the job definition revision, then manually/automatically switch the revision alias used by orchestration.
- **Network & storage parity:** Dev/staging/prod share the same VPC topology (private subnets + NAT) and EFS access-point semantics, ensuring workloads behave consistently when promoted.
- **Capacity planning:** Light compute environments cap at 256 vCPUs; heavy environments scale to 4,096 vCPUs; GPU and custom-AMI pools cap at 256 and 1,024 vCPUs respectively. Adjust Terraform variables if orchestration queues start to backlog.

Shared Conventions
------------------

- **Persistent workspace:** Every job reads/writes under `DATA_ROOT/<tour_id>` (defaults to `/mnt/shared/data/<tour_id>` on EFS). Stage output folders are reused downstream, so jobs assume the same mount across stages.
- **S3 layout:** Zipped artifacts are published to `s3://$S3_BUCKET/<user_id>/reconstruction/<tour_id>/<tour_id>.zip`. The helper `aws_unzip.py` can expand the archive back into the same prefix and delete the zip.
- **CLI arguments:**
  - Stage 1: `python entry.py <user_id> <tour_id> <is_360:0|1> <upload_type:VIDEO|IMAGE|IMAGE_GNSS|VIDEO_GNSS>`
  - Stages 2–5: `python entry.py <user_id> <tour_id>`
- **Common environment variables:**
  - `S3_BUCKET` (required): target bucket for downloads/uploads.
  - `DATA_ROOT` (optional): overrides the EFS mount point.
  - `RUN_UNZIP_IN_S3` (`true` by default): toggles the post-upload `aws_unzip.py` call.
  - `MPLCONFIGDIR`: set to `/tmp/matplotlib` in each stage to keep matplotlib happy in headless containers.
- **Optional feature flags:**
  - `RUN_DOWNLOAD` (Stage 1, default `true`): skip the initial S3 download if data is already on EFS.
  - `MASK_ENABLED` (Stage 1): also fetch masks from `reconstruction/<tour_id>/masks/`.
  - `RUN_DEPTHMAPS` (Stage 4, default `true`): compute dense depthmaps in addition to undistortion.

Stage 1 – Preprocess & Feature Extraction
-----------------------------------------

**Purpose:** bootstrap the dataset for OpenSfM by downloading raw media, generating configuration, and running the front half of the pipeline (metadata, feature extraction, matching).

- **Inputs:**
  - Raw imagery at `s3://$S3_BUCKET/<user_id>/uploads/<tour_id>/`.
  - Optional masks (when `MASK_ENABLED`): `s3://$S3_BUCKET/<user_id>/reconstruction/<tour_id>/masks/`.
  - Runtime args `user_id`, `tour_id`, `is_360`, `upload_type`.
- **Processing highlights (see `docker/stage1_preprocess/entry.py` & `reconstructor.sh`):**
  - Ensure EFS mount, set up `DATA_ROOT/<tour_id>/images`.
  - Optionally run `aws_download.py` to populate `images/` (and `masks/` if enabled).
  - Inspect the first JPEG to configure resolution flags via `configurator.py` (`--flat` skipped for 360 uploads).
  - Execute OpenSfM stages:
    1. `tools/create_exif_overrides.py` (uses `upload_type` and `user_id` to seed metadata).
    2. `extract_metadata`, `detect_features`, `match_features`.
    3. `opensfm/commands/sub_config.py` prepares submodel thresholds.
- **Outputs & uploads:**
  - Entire `DATA_ROOT/<tour_id>` directory is zipped and uploaded to S3.
  - `aws_unzip.py` (guarded by `RUN_UNZIP_IN_S3`) expands the archive back into the prefix so subsequent stages can fetch individual files.

Stage 2 – Submodel Reconstruction
---------------------------------

**Purpose:** reconstruct large tours by splitting them into submodels; skipped when the image count is small (≤300 images).

- **Inputs:**
  - Stage 1 artifacts already on EFS (`images/`, `features/`, etc.).
  - CLI args `user_id`, `tour_id`.
- **Processing (see `docker/stage2_submodels/reconstructor.sh`):**
  - Count imagery in `images/`. If >300, invoke:
    1. `bin/opensfm create_submodels`.
    2. For each `submodels/submodel_*`: `create_tracks`, `reconstruct`, `mesh`.
    3. `compute_statistics` and `export_report` per submodel.
- **Outputs & uploads:**
  - Only the `submodels/` directory is copied into a temporary folder and zipped.
  - `rot_trans_matrix_npy/` is uploaded separately if present.
  - Archive pushed to S3 and optionally unzipped in place.

Stage 3 – Full Reconstruction
-----------------------------

**Purpose:** run the main reconstruction flow (tracks → sparse reconstruction → meshing) on the full dataset when submodels are not used (or after submodels are prepared).

- **Inputs:** same as Stage 2; expects `images/` plus prior Stage 1 derivatives on EFS.
- **Processing (`docker/stage3_reconstruction/reconstructor.sh`):**
  - `bin/opensfm create_tracks`, `reconstruct`, `mesh` on the tour root (`DATA_ROOT/<tour_id>`).
- **Outputs & uploads:**
  - Zips a curated list: `tracks.csv`, `reconstruction.json`, `profile.log`, and `reports/`.
  - Uploads `rot_trans_matrix_npy/` folder if it exists.
  - Archive sent to S3 and optionally expanded via `aws_unzip.py`.

Stage 4 – Undistort & Depthmaps
-------------------------------

**Purpose:** generate undistorted imagery and (optionally) dense depthmaps for either the global model or each submodel; also aligns submodels when present.

- **Inputs:**
  - Reconstruction outputs (from Stage 2/3) available on EFS: `reconstruction.json`, `tracks.csv`, `submodels/`, etc.
  - CLI args `user_id`, `tour_id`; `RUN_DEPTHMAPS` toggle to skip depth computation.
- **Processing (`docker/stage4_undistort_depth/reconstructor.sh`):**
  - Detect if `submodels/submodel_*` directories are present.
  - For each target (submodel or full dataset):
    - `bin/opensfm undistort`.
    - `bin/opensfm compute_depthmaps` when `RUN_DEPTHMAPS=true`.
  - When submodels exist, finish with `bin/opensfm align_submodels`.
- **Outputs & uploads:**
  - Creates `DATA_ROOT/<tour_id>/<tour_id>.zip` containing:
    - `undistorted/` (excluding heavy `images/` and `depthmaps/` branches to keep size manageable).
    - For submodel runs, each `submodels/<sub>/undistorted` (with the same exclusions).
    - Top-level extras when available: `rot_trans_matrix_npy/`, `reconstruction.json`, `tracks.csv`, `merged_combined.ply`.
  - Uploads via multipart S3 transfer; optional unzip step runs afterwards.

Stage 5 – Statistics & Reporting
--------------------------------

**Purpose:** generate analytic artifacts, fix GPS/headings, and package reporting deliverables.

- **Inputs:** reconstruction outputs and imagery already on EFS.
- **Processing (`docker/stage5_stats_report/entry.py` & `reconstructor.sh`):**
  - Run `bin/opensfm compute_statistics` and `export_report`.
  - Post-processing scripts:
    - `image_headings.py`, `retrieve_missing_panos.py`, `interpolate_missing_headings.py`.
    - `split_reconstruction.py` and `tour_gps_error_handler.py` to produce GPS-corrected datasets.
  - Publish remediation notifications to the queues defined by `SQS_QUEUE_URL` and `TOUR_GPS_ERROR_HANDLER_SQS_QUEUE_URL`.
- **Outputs & uploads:**
  - Stage-only bundle is assembled in a temp folder before zipping:
    - `stats/`, `shot_headings.json`, `gps_corrected.json`, `headings_interpolated.png`, `gps_corrected_with_headings.json`, `shot_headings_with_dropped.json`, `shot_info/`.
  - Uploads the archive to the standard S3 prefix and optionally expands it with `aws_unzip.py`.

Operational Notes
-----------------

- Each stage leaves its working files in-place under EFS, so downstream jobs can resume even if S3 upload fails.
- The `aws_download.py` helper handles pagination and directory creation; it relies on the Batch role’s ability to read/write the bucket.
- When debugging locally, override `DATA_ROOT` (e.g., `/tmp/data`) and set `RUN_DOWNLOAD=false` to reuse cached imagery.
- Batch job definitions typically mount the shared EFS path, configure AWS credentials, and inject the required environment variables (`S3_BUCKET`, optionally the toggles above).
- Job definitions are revisioned; after a Terraform apply the new definition becomes `:1`, `:2`, etc. Ensure your orchestrator submits against the intended revision (or alias) before tearing down the previous one.
- Watch Batch queue metrics (runnable jobs, desired vCPUs) and EFS throughput alarms; they surface quickly when undistort/depth workloads outgrow the current limits.

Use this document as the conversational baseline when coordinating changes to the pipeline across repositories or teams. Keep environment runbooks handy for day-two operations (patch windows, incident contacts, cost tracking) that sit outside the Terraform configuration.

