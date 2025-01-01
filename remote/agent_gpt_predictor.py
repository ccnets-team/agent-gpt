# agnet_gpt_predictor.py
import json
import numpy as np
from sagemaker import Model
from remote.gpt_trainer_config import SageMakerConfig

from utils.data_converters import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays
)

class AgentGPTPredictor:
    """
    A wrapper around the SageMaker `predictor` (from `.deploy(...)`) 
    that gives user-friendly methods like `select_actions()`, 
    but under the hood calls the single SageMaker endpoint 
    with a JSON payload specifying an 'action' and 'args'.
    """
    def __init__(self, sagemaker_config: SageMakerConfig, **kwargs):
        self.model_dir: str = sagemaker_config.model_dir
        self.image_uri: str = sagemaker_config.server_uri
        self.role_arn: str = sagemaker_config.role_arn
        self.instance_count: int = sagemaker_config.instance_count
        self.instance_type: str = sagemaker_config.instance_type
        self.model_dir: str = sagemaker_config.model_dir
        self.max_run: str = sagemaker_config.max_run
        self.region: str = sagemaker_config.region

    def sagemaker_run(self):
        """
        Creates a SageMaker real-time inference endpoint using
        pre-trained model artifacts (self.model_dir) and the specified container (self.image_uri).
        """
        # 1. Create a SageMaker Model object pointing to your custom image
        model = Model(
            image_uri=self.image_uri,
            source_dir=self.model_dir,
            role=self.role_arn
        )
        print("Created SageMaker Model:", model)

        # 2. Deploy to a real-time endpoint
        self._predictor = model.deploy(
            initial_instance_count=self.instance_count,
            instance_type=self.instance_type, 
            endpoint_name="my-endpoint-name"
        )
        print("Deployed model to endpoint:", self._predictor)

        # Return self.predictor or endpoint name if you wish
        return self._predictor
    
    # -------------------------------------------------------------------------
    # Internal request helper
    # -------------------------------------------------------------------------
    def _invoke(self, action: str, args: dict) -> dict:
        """
        Helper to build a JSON payload with 'action' and 'args',
        send to the SageMaker endpoint, and parse the JSON response.
        """
        payload = {
            "action": action,
            "args": args
        }
        # Convert to JSON string
        request_str = json.dumps(payload)

        # The predictor expects either CSV, JSON, or other format
        # We'll assume JSON here if your Docker container uses JSON input
        response_bytes = self._predictor.predict(request_str.encode("utf-8"))

        # Convert from bytes -> str -> JSON
        response_str = response_bytes.decode("utf-8")
        response_data = json.loads(response_str)
        return response_data

    # -------------------------------------------------------------------------
    # Example user-friendly methods
    # -------------------------------------------------------------------------
    def select_action(self, observations: list, agent_ids: list, term_ids=None):
        """
        As in your existing client code, but calls the SageMaker endpoint
        instead of local or HTTP to a custom server.
        """
        # Convert obs to nested list if needed
        obs_converted = convert_ndarrays_to_nested_lists(observations)

        args = {
            "agent_ids": agent_ids,
            "observations": obs_converted
        }
        if term_ids is not None:
            args["term_ids"] = term_ids
        
        response = self._invoke("select_action", args)
        # Expect a structure like {"result": {"action": ...}} or {"action": ...}
        action_data = response.get("result", {}).get("action")
        if action_data is None:
            # Fallback in case it's directly in response
            action_data = response.get("action")

        # Convert to numpy if needed
        if action_data is not None:
            action_data = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        return action_data

    def reset_agents(self):
        """
        Tells the endpoint to reset all agents.
        """
        response = self._invoke("reset_agents", {})
        return response.get("result", None)

    def delete_agents(self, agent_ids: list):
        """
        Deregisters multiple agents on the endpoint.
        """
        response = self._invoke("delete_agents", {"agent_ids": agent_ids})
        return response.get("result", None)

    def set_input_seq_len(self, seq_len: int) -> bool:
        """
        Example to show how you'd pass a single param to the endpoint.
        """
        response = self._invoke("set_input_seq_len", {"seq_len": seq_len})
        success = response.get("result", None)
        return bool(success)

    def status(self):
        """
        Get the endpoint's internal status (model path, num_agents, etc.).
        """
        response = self._invoke("status", {})
        return response.get("result", None)
