# model_api.py
import numpy as np
import json
from typing import List, Optional, Union
from utils.data_converters import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
)

###############################################################################
# AgentGPTAPI: a base class that handles interaction with a SageMaker endpoint
###############################################################################

from utils.data_converters import convert_nested_lists_to_ndarrays, convert_ndarrays_to_nested_lists

class AgentGPTAPI:
    """
    AgentGPTAPI is responsible for sending actions, resets, or configuration 
    changes to the SageMaker inference endpoint, and parsing the JSON response.
    """
    def __init__(self, predictor):
        self.__predictor = predictor
        self.endpoint_name = self.__predictor.endpoint_name

    def _invoke(self, action: str, args: dict) -> dict:
        """
        Helper that builds a JSON payload, sends it to the SageMaker endpoint 
        using self.__predictor, and parses the JSON response.
        """
        payload = {"action": action, "args": args}
        request_str = json.dumps(payload)
        response_bytes = self.__predictor.predict(request_str.encode("utf-8"))
        response_str = response_bytes.decode("utf-8")
        return json.loads(response_str)

    # -------------------------------------------------------------------------
    # Example user-friendly methods (mirroring your serve.py actions)
    # -------------------------------------------------------------------------
    def select_action(
        self,
        agent_ids: Union[List[str], np.ndarray],
        observations: Union[List[np.ndarray], np.ndarray],
        terminated_agent_ids: Optional[Union[List[str], np.ndarray]] = None,
        control_values: Optional[Union[List[float], np.ndarray]] = None
    ):    
        """
        Request an action from the endpoint. 
        'observations' can be a list of np.ndarray or nested data structures.
        """
        obs_converted = convert_ndarrays_to_nested_lists(observations)
        args = {
            "agent_ids": agent_ids,
            "observations": obs_converted
        }
        if terminated_agent_ids is not None:
            args["terminated_agent_ids"] = terminated_agent_ids
        if control_values is not None:
            args["control_values"] = control_values

        response = self._invoke("select_action", args)
        # The endpoint returns something like {"action": <action array or list>}
        action_data = response.get("action")
        if action_data is not None:
            # Convert nested Python lists back into NumPy arrays
            action_data = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        return action_data

    def sample_observation(self):
        response = self._invoke("sample_observation", {})
        observation = response.get("observation", None)
        return convert_nested_lists_to_ndarrays(observation, dtype=np.float32)

    def sample_action(self):
        response = self._invoke("sample_action", {})
        action = response.get("action", None)
        return convert_nested_lists_to_ndarrays(action, dtype=np.float32)

    def get_control_value(self):
        response = self._invoke("get_control_value", {})
        control_value = response.get("control_value", None)
        return float(control_value)

    def set_control_value(self, control_value):
        return self._invoke("set_control_value", {"control_value": control_value})

    def get_max_input_states(self):
        response = self._invoke("get_max_input_states", {})
        max_input_states = response.get("max_input_states", None)
        return int(max_input_states)

    def get_num_input_states(self):
        response = self._invoke("get_num_input_states", {})
        num_input_states = response.get("num_input_states", None)
        return int(num_input_states)

    def set_num_input_states(self, num_input_states):
        return self._invoke("set_num_input_states", {"num_input_states": num_input_states})

    def terminate_agents(self, terminated_agent_ids):
        """
        Deregisters multiple agents on the endpoint by passing them
        as 'terminated_agents'.
        """
        terminated_agent_ids = convert_ndarrays_to_nested_lists(terminated_agent_ids)
        response = self._invoke("terminate_agents", {"terminated_agent_ids": terminated_agent_ids})
        return response

    def reset_agents(self, max_agents: int=None):
        """
        Tells the endpoint to reset all agents, optionally specifying a new 'max_agents'.
        """
        args = {}
        if max_agents is not None:
            args["max_agents"] = int(max_agents)
        return self._invoke("reset_agents", args)

    def status(self):
        """
        Get the endpoint's internal status (model path, num_agents, etc.).
        """
        status = self._invoke("status", {})
        return status