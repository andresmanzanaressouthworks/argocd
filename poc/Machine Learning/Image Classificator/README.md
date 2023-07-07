### Deploy an Image Classificator model on Azure ML

Pre-requisites:

- An Azure ML workspace.
- A compute cluster in the workspace.

Example of use:

``` python deploy_and_run.py --compute_cluster "gpu-cluster" --subscription_id "d05a822f-be11-493a-8fac-9030fa6c2a4a" --resource_group "machine-learning" --workspace "mlworkspace"```

Based on the tutorial: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-image-processing-batch?view=azureml-api-2&tabs=python