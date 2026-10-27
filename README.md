# gl-automobilevehiclepricing

## Model objective

Predict fair vehicle prices from specifications to replace slow, error-prone manual dealership pricing.

The pipeline cleans and transforms vehicle segment, distance, mileage, engine, power,
and seating data; trains and evaluates a Random Forest price model; logs MLflow
metrics; and registers the selected model in Azure Machine Learning.

## Azure MLflow workflows

The following GitHub Actions workflows appear in the repository's **Actions** tab:

- `create-compute` creates or updates the `cpu-cluster` Azure ML compute target.
- `register-environment` registers the training environment used by the pipeline.
- `deploy-model-training-pipeline` registers the dataset, validates the pipeline, and
  registers its training component.
- `run-pipeline` submits the Azure ML pipeline, where MLflow records training parameters,
  metrics, and the registered model.

Before running a workflow, configure these repository settings:

| Setting | Purpose |
|---|---|
| Secret `AZURE_CREDENTIALS` | Service-principal JSON accepted by [`azure/login`](https://github.com/Azure/login). |

The service principal must also have Azure RBAC access to the resource group or ML workspace;
repository access alone does not grant Azure permissions.

Run the workflows in this order: `register-environment`, `create-compute`,
`deploy-model-training-pipeline`, then `run-pipeline`.
