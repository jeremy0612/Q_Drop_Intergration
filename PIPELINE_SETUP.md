# CML + DVC Setup for Q-Drop Training

## Prerequisites

1. **GitHub runner with A6000 tag**
   ```bash
   # On A6000 machine, register runner:
   ./config.sh --url https://github.com/YOUR_REPO --token GHPR_xxx --labels gpu,A6000
   # Keep runner service running (systemd/launchd)
   ```

2. **S3 remote for model storage (optional but recommended)**
   ```bash
   dvc remote add -d myremote s3://your-bucket/q-drop-models
   dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
   dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY
   ```

3. **GitHub token for PR comments**
   - Add `REPO_TOKEN` via GitHub Actions secrets (auto-inherited from `${{ secrets.GITHUB_TOKEN }}`)

## Workflow

**On Push to main:**
- Train all 3 algorithms (pruning, dropout, combined)
- Save metrics.json
- Push results to GitHub
- Update baseline.json
- DVC push models to S3

**On PR:**
- Train pipeline
- Compare metrics vs baseline
- Post result as PR comment with plot

**Files Added:**
- `.github/workflows/train.yml` — CI pipeline
- `dvc.yaml` — DVC pipeline config
- `params.yaml` — hyperparameters (tracked)
- `baseline.json` — baseline metrics
- Modified `src/train_mnist.py` — JSON output

## Run Locally

Test pipeline before pushing:
```bash
dvc repro
# Check metrics.json
cat metrics.json
```

## Monitoring

- Watch runs: GitHub Actions tab
- View metrics: Pull request comments
- Access models: S3 bucket (DVC managed)
