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

from .config.sagemaker import SageMakerConfig
from .config.hyperparams import Hyperparameters
from .gpt_api import GPTAPI

class AgentGPT:
    """
    AgentGPT is your one‑click solution for **training** and **running** a 
    multi‑agent RL model on AWS SageMaker. This class provides:
    
      1) **train**: Launch a training job in SageMaker.
      2) **infer**: Deploy a real‑time inference endpoint in SageMaker.
      3) **Return a GPTAPI client** to communicate with the deployed model 
         (for actions, control values, etc.).

    Note on Environment Hosting:
    ----------------------------
    While AgentGPT coordinates the RL model’s training and inference,
    it **does not** manage environment hosting (simulation) itself. 
    That is assumed to be set up separately—either locally or in the cloud—
    using tools in the **`env_host`** directory. 
    The environment server (e.g. a FastAPI app) should already be accessible 
    by the time you run `train` or `infer`.

    Container Image URI:
    ----------------------
    The image URI is generated dynamically using `get_image_uri()`,  
    which builds the URI based on the configured region and service type  
    ("trainer" or "inference"). This ensures that the image used aligns  
    with the user's region and settings.
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

        This method packages up your environment, hyperparameters, and the dynamically
        computed Docker image reference (using `get_image_uri("trainer")` from the 
        SageMakerConfig) into a SageMaker Estimator. Then it calls `estimator.fit()` 
        to run a cloud-based training job.

        **Usage Example**::

            from agent_gpt import AgentGPT
            from src.config.aws_config import SageMakerConfig
            from src.config.hyperparams import Hyperparameters

            sagemaker_cfg = SageMakerConfig(...)
            hyperparams = Hyperparameters(...)

            # Kick off training in the cloud
            estimator = AgentGPT.train(sagemaker_cfg, hyperparams)
            print("Training job submitted:", estimator.latest_training_job.name)

        :param sagemaker_config: 
            A SageMakerConfig containing details like `role_arn`, `region`, and other 
            configuration settings. The container image URI is computed via 
            `sagemaker_config.get_image_uri("trainer")`.
        :param hyperparameters:
            A Hyperparameters object with fields needed to configure environment, RL training, 
            and additional settings.
        :return:
            A `sagemaker.estimator.Estimator` instance that has started 
            the training job. You can query `.latest_training_job` for status.
        """
        
        _validate_sagemaker(sagemaker_config)
        _validate_hyperparams(hyperparameters)

        trainer_config = sagemaker_config.trainer
        
        # Check for default output_path
        if trainer_config.output_path == trainer_config.DEFAULT_OUTPUT_PATH:
            raise ValueError("Invalid output_path: Please update the SageMaker trainer output_path to a valid S3 location.")
        
        image_uri = sagemaker_config.get_image_uri("trainer")
        hyperparams_dict = hyperparameters.to_dict()
        estimator = Estimator(
            image_uri=image_uri,
            role=sagemaker_config.role_arn,
            instance_type=trainer_config.instance_type,
            instance_count=trainer_config.instance_count,
            output_path=trainer_config.output_path,
            max_run=trainer_config.max_run,
            region=sagemaker_config.region,
            hyperparameters=hyperparams_dict
        )
        estimator.fit()
        return estimator
        
    @staticmethod
    def infer(sagemaker_config: SageMakerConfig):
        """
        Creates (or reuses) a SageMaker real-time inference endpoint for AgentGPT.

        This method uses your pre-trained model artifacts, and the container image URI
        is determined dynamically by calling `get_image_uri("inference")` on the SageMakerConfig.
        It uses other configuration details (such as model_data, instance type, and endpoint name)
        from the SageMakerConfig to build and/or deploy a SageMaker Endpoint. The `endpoint_name`
        field is used to determine the name of the deployed inference endpoint. If not provided,
        a default name is auto-generated.

        Workflow:
          1) A `Model` object is created referencing your model data in S3 (e.g. `model_data`)
             and the container image (computed via `get_image_uri("inference")`).
          2) The method checks if an endpoint with the specified `endpoint_name` already exists.
          3) If it exists, the existing endpoint is reused by creating a `Predictor`.
          4) Otherwise, the method calls `.deploy(...)` on the `Model` to create a new endpoint.
          5) Finally, a GPTAPI object is returned to communicate with the deployed endpoint.

        :param sagemaker_config:
            Contains the AWS IAM role, model data path, instance type, and the 
            `endpoint_name` to be used for the deployed inference endpoint. The container image 
            is computed using `sagemaker_config.get_image_uri("inference")`.
        :return:
            A `GPTAPI` instance, preconfigured to call the SageMaker endpoint for inference.
        """
        inference_config = sagemaker_config.inference
        # Check for default model_data
        if inference_config.model_data == inference_config.DEFAULT_MODEL_DATA:
            raise ValueError("Invalid model_data: Please update the SageMaker inference model_data to a valid S3 location.")
        
        image_uri = sagemaker_config.get_image_uri("inference")
        model = Model(
            role=sagemaker_config.role_arn,
            image_uri=image_uri,
            model_data=inference_config.model_data
        )
        print("Created SageMaker Model:", model)
        
        endpoint_name = inference_config.endpoint_name

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
                initial_instance_count=inference_config.instance_count,
                instance_type=inference_config.instance_type,
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
    Validate required hyperparameters (env_id, env_hosts, etc.).
    Raises ValueError if any required parameter is missing or invalid.
    """
    if params.env_id is None:
        raise ValueError("Environment ID must be provided.")
    if not params.env_hosts:
        raise ValueError("No environment hosts specified in 'env_hosts'. "
                         "Please run 'agent-gpt simulate ...' first to set up an environment.")
    # If entry_point is provided, normalize it.
    if params.env_entry_point:
        import os
        params.env_entry_point = os.path.dirname(os.path.abspath(params.env_entry_point)).replace(os.sep, "/")
