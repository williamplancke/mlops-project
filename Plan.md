# Commands 
Start: docker run --rm -v azure-cli-config:/root/.azure mcr.microsoft.com/azure-cli az aks start --resource-group azure-ai --name mlops-project-aks
Status: docker run --rm -v azure-cli-config:/root/.azure mcr.microsoft.com/azure-cli az aks show --resource-group azure-ai --name mlops-project-aks --query powerState
Stop: docker run --rm -v azure-cli-config:/root/.azure mcr.microsoft.com/azure-cli az aks stop --resource-group azure-ai --name mlops-project-aks
Delete: docker run --rm -v azure-cli-config:/root/.azure mcr.microsoft.com/azure-cli az aks delete --resource-group azure-ai --name mlops-project-aks --yes



# MLOps Deployment Plan

This project deploys three customer models trained from `train_V2_cleaned.csv`:

- `outcome_profit`: expected customer profit.
- `outcome_damage_inc`: probability that the customer causes damage.
- `outcome_damage_amount`: expected damage amount if damage happens.

## Task 1: Cloud AI Training

Training is implemented as a repeatable Azure ML command job:

- Local/cloud entrypoint: `training/train.py`
- Azure ML job spec: `training/azure_job.yml`
- Azure ML environment: `training/conda.yml`

The training job writes four artifacts to `models/`:

- `profit_model.joblib`
- `damage_incidence_model.joblib`
- `damage_amount_model.joblib`
- `model_metadata.json`

For demonstration runs, the Azure job uses `--quick` to reduce training time and cloud cost while preserving the same model structure.

## Task 2: Kubernetes Deployment

The Kubernetes deployment is in `k8s/` and contains:

- FastAPI model service: `api/main.py`
- Frontend: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`
- Persistent PostgreSQL database: `k8s/postgres.yml`
- Production NGINX reverse proxy: `nginx/default.conf`, `k8s/nginx.yml`

Request flow:

1. User opens the NGINX LoadBalancer service.
2. NGINX routes `/` to the frontend service.
3. NGINX routes `/api/*` to the FastAPI service.
4. FastAPI loads the three trained model artifacts and logs every prediction to PostgreSQL.

Apply manually with:

```powershell
kubectl apply -k k8s
```

Before applying outside CI, replace `ghcr.io/OWNER/REPO` in `k8s/*.yml` with your actual image prefix.

## Task 3: CI/CD

Three GitHub Actions workflows implement the required lifecycle:

- Task 3.1: `.github/workflows/retrain-model.yml`
  - Triggers when training code, shared model metadata, or the training dataset changes.
  - Creates the Azure ML workspace and autoscaling CPU compute when missing.
  - Submits and monitors the Azure ML training job.
  - Validates and registers the resulting model in Azure ML.
  - Scales the Azure ML compute cluster back to zero even when a prior step fails.

- Task 3.2: `.github/workflows/deploy-model.yml`
  - Triggers automatically after the retraining workflow completes successfully.
  - Downloads the latest registered Azure ML model.
  - Builds and publishes a new API image containing that model.
  - Updates only the API deployment using Kubernetes `RollingUpdate`.
  - Waits for the rollout to become healthy.

- Task 3.3: `.github/workflows/deploy-application.yml`
  - Triggers when API, frontend, NGINX, Compose, dataset-ranking, or Kubernetes configuration changes.
  - Restores the latest registered model for the API image.
  - Rebuilds and publishes all application images.
  - Applies the Kubernetes manifests.
  - Performs rolling updates for API, frontend, and NGINX.

The two deployment workflows share a concurrency group, preventing simultaneous
updates to the AKS cluster.

Required GitHub secrets:

- `AZURE_CLIENT_ID`: Azure service principal or managed identity client ID.
- `AZURE_TENANT_ID`: Azure tenant ID.
- `AZURE_SUBSCRIPTION_ID`: Azure subscription ID.
- `AZURE_RESOURCE_GROUP`: Azure resource group name.
- `AZURE_LOCATION`: Azure region, for example `westeurope`.
- `AZURE_WORKSPACE`: Azure ML workspace name.
- `AKS_CLUSTER_NAME`: AKS cluster name.

The workflows use Azure Login with OIDC, so the Azure identity must have a federated credential for this GitHub repository and branch. GitHub Container Registry uses the built-in `GITHUB_TOKEN`.

## Local Verification

Create a local environment and run a fast training pass:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe training/train.py --data train_V2_cleaned.csv --output-dir models --quick
```

Run the API locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open the frontend through the Kubernetes/NGINX deployment, or call the API directly at `http://127.0.0.1:8000/predict`.
