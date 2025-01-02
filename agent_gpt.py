import json
import numpy as np
import re

from sagemaker import Model
from sagemaker.estimator import Estimator

from config.aws_config import SageMakerConfig
from config.hyperparams import Hyperparameters
from utils.data_converters import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays
)

###############################################################################
# AgentGPTAPI: a base class that handles interaction with a SageMaker endpoint
###############################################################################
class AgentGPTAPI:
    """
    AgentGPTAPI is responsible for sending actions, resets, or configuration 
    changes to the SageMaker inference endpoint, and parsing the JSON response.
    """
    def __init__(self):
        self._predictor = None
    
    def initalize(self, predictor):
        """
        Initialize this class with a sagemaker.Predictor object.
        """
        self._predictor = predictor

    def _invoke(self, action: str, args: dict) -> dict:
        """
        Helper that builds a JSON payload, sends it to the SageMaker endpoint 
        using self._predictor, and parses the JSON response.
        """
        payload = {"action": action, "args": args}
        request_str = json.dumps(payload)
        response_bytes = self._predictor.predict(request_str.encode("utf-8"))
        response_str = response_bytes.decode("utf-8")
        response_data = json.loads(response_str)
        return response_data

    # -------------------------------------------------------------------------
    # Example user-friendly methods
    # -------------------------------------------------------------------------
    def select_action(self, observations: list, agent_ids: list, term_ids=None):
        obs_converted = convert_ndarrays_to_nested_lists(observations)
        args = {"agent_ids": agent_ids, "observations": obs_converted}
        if term_ids is not None:
            args["term_ids"] = term_ids
        
        response = self._invoke("select_action", args)
        action_data = response.get("result", {}).get("action") or response.get("action")
        if action_data is not None:
            action_data = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        return action_data

    def reset_agents(self):
        """Tells the endpoint to reset all agents."""
        response = self._invoke("reset_agents", {})
        return response.get("result", None)

    def delete_agents(self, agent_ids: list):
        """Deregisters multiple agents on the endpoint."""
        response = self._invoke("delete_agents", {"agent_ids": agent_ids})
        return response.get("result", None)

    def set_input_seq_len(self, seq_len: int) -> bool:
        """Example to show how you'd pass a single param to the endpoint."""
        response = self._invoke("set_input_seq_len", {"seq_len": seq_len})
        success = response.get("result", None)
        return bool(success)

    def status(self):
        """
        Get the endpoint's internal status (model path, num_agents, etc.).
        """
        response = self._invoke("status", {})
        return response.get("result", None)


###############################################################################
# AgentGPT: the main class for training and running the RL environment in SageMaker
###############################################################################
class AgentGPT(AgentGPTAPI):
    """
    AgentGPT extends AgentGPTAPI with methods to perform SageMaker training 
    and deployment, as well as any other high-level logic.
    """

    def __init__(self):
        super().__init__()
        self._estimator = None
        self._predictor = None

    def train_on_cloud(self, sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """
        Launch a SageMaker training job for a one-click robotics environment.
        """
        _validate_sagemaker(sagemaker_config)
        _validate_hyperparams(hyperparameters)

        hyperparameters.model_dir = sagemaker_config.model_dir
        if hyperparameters.env_tag is not None:
            hyperparameters.env_id += hyperparameters.env_tag

        hyperparams_dict = hyperparameters.to_dict()
        self._estimator = Estimator(
            image_uri=sagemaker_config.trainer_uri,
            role=sagemaker_config.role_arn,
            instance_type=sagemaker_config.instance_type,
            instance_count=sagemaker_config.instance_count,
            output_path=sagemaker_config.model_dir,
            max_run=sagemaker_config.max_run,
            region=sagemaker_config.region,
            hyperparameters=hyperparams_dict
        )
        self._estimator.fit()
        
    def act_on_cloud(self, sagemaker_config: SageMakerConfig):
        """
        Creates a SageMaker real-time inference endpoint using pre-trained 
        model artifacts (self.model_dir) and the specified container (self.image_uri).
        """
        model = Model(
            image_uri=sagemaker_config.server_uri,
            model_data=sagemaker_config.model_dir + "model.tar.gz",
            role=sagemaker_config.role_arn
        )
        print("Created SageMaker Model:", model)

        self._predictor = model.deploy(
            initial_instance_count=sagemaker_config.instance_count,
            instance_type=sagemaker_config.instance_type, 
            endpoint_name="my-endpoint-name"
        )
        print("Deployed model to endpoint:", self._predictor)
        
        # Now initialize the base class with the new predictor
        super().initalize(self._predictor)


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
    if params.env_url is None:
        raise ValueError("Must provide an environment URL.")
