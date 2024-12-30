import json
import numpy as np

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

    def __init__(self, predictor):
        """
        :param predictor: The sagemaker RealTimePredictor or Predictor object
                          from `self.inference_model.deploy(...)`.
        """
        self._predictor = predictor
    
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
    def select_action(self, agent_ids: list, observations: list, term_ids=None, control_values=None):
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
        if control_values is not None:
            args["control_values"] = control_values
        
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

    # ... You can replicate more methods like set_global_control_value, etc.

