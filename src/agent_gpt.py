# src/agent_gpt.py
###############################################################################
# AgentGPT: the main class for training and running an RL environment in SageMaker
###############################################################################
import re
import time
import boto3
from sagemaker import Model
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor

from src.config.sagemaker_config import SageMakerConfig
from src.config.hyperparams import Hyperparameters
from src.gpt_api import GPTAPI

class AgentGPT:
    """
    AgentGPT is your one-click solution for **training** and **running** a 
    multi-agent RL model on AWS SageMaker. This class provides:
    
      1) **train_on_cloud**: Launch a training job in SageMaker.
      2) **run_on_cloud**: Deploy a real-time inference endpoint in SageMaker.
      3) **return a GPTAPI client** to communicate with the deployed model 
         (for actions, control values, etc.).

    Note on Environment Hosting:
    ----------------------------
    While AgentGPT coordinates the RL model’s training and inference,
    it **does not** manage environment hosting (simulation) itself. 
    That is assumed to be set up separately—either locally or in the cloud—
    using tools in the **`env_host`** directory. 
    The environment server (e.g. a FastAPI app) should already be accessible 
    by the time you run `train_on_cloud` or `run_on_cloud`.

    Why Static Methods?
    -------------------
    We keep `train_on_cloud` and `run_on_cloud` as static methods to emphasize
    “AgentGPT” as the central orchestrator for your RL workflows—no instantiation
    required. You simply call `AgentGPT.train_on_cloud(...)` or
    `AgentGPT.run_on_cloud(...)` to handle everything from code packaging 
    to launching the SageMaker job or endpoint.
    """

    def __init__(self):
        """
        Currently unused as AgentGPT only has static methods.
        """
        pass
        
    @staticmethod
    def train(sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """
        Launch a SageMaker training job for your AgentGPT environment.

        This method packages up your environment, hyperparameters, 
        and Docker image references (from sagemaker_config) into a 
        SageMaker Estimator. Then it calls `estimator.fit()` to run
        a cloud-based training job.

        **Usage Example**::
        
            from agent_gpt import AgentGPT
            from src.config.aws_config import SageMakerConfig
            from src.config.hyperparams import Hyperparameters

            sagemaker_cfg = SageMakerConfig(...)
            hyperparams = Hyperparameters(...)

            # Kick off training in the cloud
            estimator = AgentGPT.train_on_cloud(sagemaker_cfg, hyperparams)
            print("Training job submitted:", estimator.latest_training_job.name)
        
        :param sagemaker_config: 
            A SageMakerConfig containing details like `role_arn`, 
            `image_uri`, `instance_type`, etc.
        :param hyperparameters:
            A Hyperparameters object with fields needed to configure
            environment, RL training, and additional settings.
        :return:
            A `sagemaker.estimator.Estimator` instance that has started 
            the training job. You can query `.latest_training_job` for status.
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
    def infer(sagemaker_config: SageMakerConfig):
        """
        Creates (or reuses) a SageMaker real-time inference endpoint for AgentGPT.

        This method uses your pre-trained model artifacts, the container image (`image_uri`),
        and other configuration details from `sagemaker_config` to build and/or deploy a SageMaker Endpoint.
        The `endpoint_name` field in the configuration is used to determine the name of the deployed
        inference endpoint. If not provided, a default name is auto-generated.

        Workflow:
          1) A `Model` object is created referencing your model data in S3 (e.g. `model_data`)
             and the container image (`image_uri`).
          2) The method checks if an endpoint with the specified `endpoint_name` already exists.
          3) If it exists, the existing endpoint is reused by creating a `Predictor`.
          4) Otherwise, the method calls `.deploy(...)` on the `Model` to create a new endpoint.
          5) Finally, a GPTAPI object is returned to communicate with the deployed endpoint.

        :param sagemaker_config:
            Contains the AWS IAM role, model data path, instance type, and the 
            `endpoint_name` to be used for the deployed inference endpoint.
        :return:
            A `GPTAPI` instance, preconfigured to call the SageMaker endpoint for inference.
        """
        model = Model(
            role=sagemaker_config.role_arn,
            image_uri=sagemaker_config.image_uri,
            model_data=sagemaker_config.model_data
        )
        print("Created SageMaker Model:", model)
        
        endpoint_name = sagemaker_config.endpoint_name

        if not endpoint_name:
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
            print(f"Reusing existing endpoint: {endpoint_name}")
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=model.sagemaker_session
            )
        else:
            print(f"Creating a new endpoint: {endpoint_name}")
            new_predictor = model.deploy(
                initial_instance_count=sagemaker_config.instance_count,
                instance_type=sagemaker_config.instance_type,
                endpoint_name=endpoint_name
            )
            print("Deployed model to endpoint:", new_predictor)
            
            if new_predictor is not None:
                predictor = new_predictor
            else:
                print("model.deploy(...) returned None, creating Predictor manually...")
                predictor = Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=model.sagemaker_session
                )    

        # Return a GPTAPI client for inference calls
        return GPTAPI(predictor)


def _validate_sagemaker(sagemaker_config: SageMakerConfig):
    """
    Validate essential fields in SageMakerConfig (role ARN, etc.).
    Raises ValueError if invalid.
    """
    if (not sagemaker_config.role_arn or
            not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sagemaker_config.role_arn)):
        print("Role ARN:", sagemaker_config.role_arn)
        raise ValueError("Must provide a valid AWS IAM Role ARN.")

def _validate_hyperparams(params: Hyperparameters):
    """
    Validate required hyperparameters (env_id, env_endpoint, etc.).
    Raises ValueError if missing or invalid.
    """
    if params.env_id is None:
        raise ValueError("Must provide an environment ID.")
    if params.env_hosts is None:
        raise ValueError("Must provide an environment URL.")
