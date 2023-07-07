from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import BatchEndpoint, Data, ModelBatchDeployment, Model, BatchRetrySettings, \
	CodeConfiguration, \
	Environment, ResourceConfiguration, ModelBatchDeploymentSettings
import os
import urllib.request
from zipfile import ZipFile
import json


def create_batch_endpoint(ml_client):
	print("Creating Batch Endpoint...")
	endpoint = BatchEndpoint(
		name="imagenet-classifier-batch",
		description="An batch service to perform ImageNet image classification",
	)
	ml_client.batch_endpoints.begin_create_or_update(endpoint)
	print("Batch Endpoint created.")
	return endpoint


def get_environment():
	return Environment(
		name="tensorflow27-cuda11-gpu",
		conda_file="environment/conda.yml",
		image="mcr.microsoft.com/azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu:latest",
	)


def download_and_register_model(ml_client):
	print("Downloading model...")
	response = urllib.request.urlretrieve(
		'https://azuremlexampledata.blob.core.windows.net/data/imagenet/model.zip',
		'model.zip'
	)
	model_path = "imagenet-classifier"
	os.makedirs(model_path, exist_ok=True)
	print("Extracting model...")
	with ZipFile(response[0], 'r') as model_zip:
		model_zip.extractall(path=model_path)
	model_name = 'imagenet-classifier'
	print("Uploading model...")
	model = ml_client.models.create_or_update(
		Model(name=model_name, path=model_path, type=AssetTypes.CUSTOM_MODEL)
	)
	print("Model uploaded.")
	return model


def download_and_register_validation_data(ml_client):
	print("Downloading dataset...")
	response = urllib.request.urlretrieve(
		'https://azuremlexampledata.blob.core.windows.net/data/imagenet/imagenet-1000.zip',
		'imagenet-1000.zip'
	)
	data_path = "data"
	os.makedirs(data_path, exist_ok=True)
	print("Extracting dataset...")
	with ZipFile(response[0], 'r') as dataset_zip:
		dataset_zip.extractall(path=data_path)
	dataset_name = "imagenet-sample-unlabeled"
	imagenet_sample = Data(
		path=data_path,
		type=AssetTypes.URI_FOLDER,
		description="A sample of 1000 images from the original ImageNet dataset",
		name=dataset_name,
	)
	print("Uploading dataset...")
	ml_client.data.create_or_update(imagenet_sample)
	print("Dataset uploaded.")
	return ml_client.data.get(dataset_name, label="latest")


def create_deployment(compute_name, endpoint, environment, model, ml_client):
	deployment = ModelBatchDeployment(
		name="imagenet-classifier-resnetv2",
		description="A ResNetV2 model architecture for performing ImageNet classification in batch",
		endpoint_name=endpoint.name,
		model=model,
		environment=environment,
		code_configuration=CodeConfiguration(
			code="code/score-by-file",
			scoring_script="batch_driver.py",
		),
		compute=compute_name,
		resources=ResourceConfiguration(instance_count=1),
		settings=ModelBatchDeploymentSettings(
			max_concurrency_per_instance=1,
			mini_batch_size=10,
			output_action=BatchDeploymentOutputAction.APPEND_ROW,
			output_file_name="predictions.csv",
			retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
			logging_level="info"
		)
	)
	print("Creating deployment...")
	ml_client.batch_deployments.begin_create_or_update(deployment)
	print("Deployment created.")


def get_imagenet_labels():
	with open('labels.json') as labels_file:
		labels = json.load(labels_file)
	return labels


def invoke_endpoint(data_asset_for_testing, endpoint, ml_client):
	job_input = Input(type=AssetTypes.URI_FOLDER, path=data_asset_for_testing.id)
	print("Invoking endpoint...")
	job = ml_client.batch_endpoints.invoke(
		deployment_name="imagenet-classifier-resnetv2",
		endpoint_name=endpoint.name,
		input=job_input,
	)
	job_status = ml_client.jobs.get(job.name).status
	while job_status != "Completed" and job_status != "Failed":
		job_status = ml_client.jobs.get(job.name).status
	if job_status == "Failed":
		raise Exception("The image processing failed. Please check the logs")
	ml_client.jobs.download(name=job.name, output_name='score', download_path='./')
	print("Endpoint finished processing.")


def get_results():
	import pandas as pd
	score = pd.read_csv("predictions.csv", header=None, names=['file', 'class', 'probabilities'], sep=' ')
	imagenet_labels = get_imagenet_labels()
	score['label'] = score['class'].apply(lambda pred: imagenet_labels[str(pred - 1)])
	return score
