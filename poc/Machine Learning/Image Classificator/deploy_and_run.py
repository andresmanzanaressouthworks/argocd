from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from aux_functions import *
import argparse
from IPython.core.display_functions import display


def deploy(compute_cluster, ml_client):
	endpoint = create_batch_endpoint(ml_client)
	model = download_and_register_model(ml_client)
	environment = get_environment()
	create_deployment(compute_cluster, endpoint, environment, model, ml_client)
	return endpoint


def run(endpoint, ml_client):
	data_asset_for_testing = download_and_register_validation_data(ml_client)
	invoke_endpoint(data_asset_for_testing, endpoint, ml_client)
	score_df = get_results()
	display(score_df)


def main(compute_cluster, subscription_id, resource_group, workspace):
	ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
	endpoint = deploy(compute_cluster, ml_client)
	run(endpoint, ml_client)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Script to deploy and run an Image Classifier on Azure ML .")

	parser.add_argument("--compute_cluster", help="Compute Cluster to run the model", required=True)
	parser.add_argument("--subscription_id", help="Subscription ID of the Azure ML workspace", required=True)
	parser.add_argument("--resource_group", help="Resource Group of the Azure ML workspace", required=True)
	parser.add_argument("--workspace", help="Name of the Azure ML workspace", required=True)

	args = parser.parse_args()

	main(args.compute_cluster, args.subscription_id, args.resource_group, args.workspace)
