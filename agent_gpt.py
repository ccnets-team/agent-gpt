# agent_gpt.py
###############################################################################
# AgentGPT: the main class for training and running the RL environment in SageMaker
###############################################################################
import re
import time
import boto3
from sagemaker import Model
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor

from config.aws_config import SageMakerConfig
from config.hyperparams import Hyperparameters
from gpt_api import GPTAPI

class AgentGPT:
    def __init__(self):
        pass
        
    @staticmethod
    def train_on_cloud(sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """
        Launch a SageMaker training job for a one-click robotics environment.
        """
        _validate_sagemaker(sagemaker_config)
        _validate_hyperparams(hyperparameters)

        hyperparams_dict = hyperparameters.to_dict()
        
        estimator = Estimator(
            image_uri=sagemaker_config.image_uri,
            role=sagemaker_config.role_arn,
            instance_type=sagemaker_config.instance_type,
            instance_count=sagemaker_config.instance_count,
            output_path=sagemaker_config.output_path,
            max_run=sagemaker_config.max_run,
            region=sagemaker_config.region,
            hyperparameters=hyperparams_dict
        )
        estimator.fit()
        
        return estimator
        
    @staticmethod
    def run_on_cloud(sagemaker_config, user_endpoint_name: str = None):
        """
        Creates or reuses a SageMaker real-time inference endpoint using pre-trained 
        model artifacts and the specified container.
        """
        model = Model(
            role=sagemaker_config.role_arn,
            image_uri=sagemaker_config.image_uri,
            model_data=sagemaker_config.model_data
        )
        print("Created SageMaker Model:", model)

        if user_endpoint_name:
            endpoint_name = user_endpoint_name
        else:
            endpoint_name = f"agent-gpt-{int(time.time())}"
            
        print("Using endpoint name:", endpoint_name)

        sagemaker_client = boto3.client("sagemaker")
        try:
            desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = desc["EndpointStatus"]
            print(f"Endpoint '{endpoint_name}' exists (status: {endpoint_status}).")
            endpoint_exists = True
        except sagemaker_client.exceptions.ClientError:
            endpoint_exists = False

        if endpoint_exists:
            # Reuse the existing endpoint
            print(f"Reusing existing endpoint: {endpoint_name}")
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=model.sagemaker_session
            )
        else:
            # Deploy a new endpoint
            print(f"Creating a new endpoint: {endpoint_name}")

            new_predictor = model.deploy(
                initial_instance_count=sagemaker_config.instance_count,
                instance_type=sagemaker_config.instance_type,
                endpoint_name=endpoint_name
            )
            print("Deployed model to endpoint:", new_predictor)
            
            if new_predictor is not None:
                # The SDK returned a valid Predictor object
                predictor = new_predictor
            else:
                # If it somehow returned None, create the predictor manually
                print("model.deploy(...) returned None, creating Predictor manually...")
                predictor = Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=model.sagemaker_session
                )    

        # Initialize your API or parent class
        return GPTAPI(predictor)

def _validate_sagemaker(sagemaker_config: SageMakerConfig):
    """
    Validate the SageMaker training job configuration.
    """
    if (not sagemaker_config.role_arn or
            not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sagemaker_config.role_arn)):
        print("Role ARN:", sagemaker_config.role_arn)
        raise ValueError("Must provide a valid AWS IAM Role ARN.")

def _validate_hyperparams(params: Hyperparameters):
    """
    Validate the one-click hyperparameters for environment setup or training.
    """
    if params.env_id is None:
        raise ValueError("Must provide an environment ID.")
    if params.env_endpoint is None:
        raise ValueError("Must provide an environment URL.")
