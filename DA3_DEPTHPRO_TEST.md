# DA3 DepthPro Replacement Test

Quick test setup to evaluate Depth Anything 3 (DA3) as a replacement for Apple DepthPro in AWS Batch SfM pipeline.

## Goal

Test if DA3 can fix seam/consistency issues in equirectangular depth maps caused by DepthPro processing cubemap faces independently.

## Setup

```bash
# 1. Copy stitching script from your depthpro_docker_image repo
cp ../depthpro_docker_image/stitch_depth_equirect_parallel.py ./

# 2. Build Docker image
docker build -t depthpro-da3 .

# 3. Push to ECR
docker tag depthpro-da3 YOUR_ECR_REGISTRY/depthpro_da3_dev:latest
docker push YOUR_ECR_REGISTRY/depthpro_da3_dev:latest
```

## What Changed vs DepthPro

| Component | DepthPro | DA3 (This Test) |
|-----------|----------|-----------------|
| **Model** | apple/DepthPro-hf (~300M) | depth-anything/DA3METRIC-LARGE (350M) |
| **Output** | Metric depth | Metric depth + confidence + sky seg |
| **Entrypoint** | entrypoint.py | entrypoint_da3.py (identical logic) |
| **Inference** | depthpro_batch.py | da3_batch.py (mirrors structure) |
| **Stitching** | ✅ Same file | ✅ Same file |
| **Infrastructure** | g4dn.xlarge | ✅ Same (g4dn.xlarge) |
| **Interface** | `python entrypoint.py USER TOUR BATCH` | ✅ Same interface |

## Run

### AWS Batch
Same command as DepthPro, just use different job definition:
```bash
aws batch submit-job \
  --job-name "da3-test-tour-XXXXX" \
  --job-queue "depthpro-queue_dev" \
  --job-definition "depthpro_da3-definition_dev:1" \
  --array-properties size=10 \
  --parameters user_id=10477,tour_id=410704,batch_size=1
```

### Local Test (Optional)
```bash
export S3_BUCKET=your-bucket
export DATA_ROOT=/mnt/shared/data
export AWS_BATCH_JOB_ARRAY_INDEX=0

python entrypoint_da3.py 10477 410704 1
```

## Compare Results

1. Run DepthPro job → Download outputs from S3
2. Run DA3 job on same tour → Download outputs from S3  
3. Visually compare equirectangular depth maps
4. Look for improvements in:
   - Seam quality at face boundaries
   - Depth consistency across panorama
   - Sky region handling

Expected: Better seam quality, fewer depth discontinuities.

## Files

- `Dockerfile` - Mirrors DepthPro Dockerfile structure
- `entrypoint_da3.py` - Mirrors DepthPro entrypoint.py logic
- `da3_batch.py` - Mirrors depthpro_batch.py, uses DA3 model
- `stitch_depth_equirect_parallel.py` - Your existing stitching script (copied)

## Model Choice: DA3METRIC-LARGE

Why this model:
- ✅ Similar size to DepthPro (350M vs 300M params)
- ✅ Metric depth output (same as DepthPro)
- ✅ Fits on g4dn.xlarge (16GB GPU)
- ✅ Apache 2.0 license (production ready)
- ✅ Built-in sky segmentation (bonus)

Alternative if seams still problematic:
- DA3NESTED-GIANT-LARGE (1.4B params) with multi-view mode
- Requires g4dn.2xlarge (32GB GPU)
- Processes all 6 faces together with geometric constraints

## Notes

- This is single-view mode (like DepthPro - each face processed independently)
- Improvement expected from better depth prediction model, not multi-view consistency
- If seams improve significantly → Success! 
- If seams still bad → Need multi-view mode (larger infrastructure)

Good luck! 🤞

