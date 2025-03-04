# gpt_api.py
import numpy as np
import json
from typing import List, Optional, Union, Dict
from .utils.conversion_utils import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
)

###############################################################################
# GPTAPI: a base class that handles interaction with a SageMaker endpoint
###############################################################################

class GPTAPI:
    """
    GPTAPI handles communication with a SageMaker inference endpoint for
    multi-agent environments, including:

    1) **GPT Input Sequence Length (num_input_states)**:
       - This controls the 'context window' size for GPT-based models. 
         A higher number of input states means the model sees more 
         historical steps (observations, actions) when selecting an action.
       - Currently, all agents share the same num_input_states globally,
         though future updates may allow per-agent customization.

    2) **Agent Performance Control Value** (in [0.0, 1.0]):
       - This is a scalar (or array of scalars) modulating how 'capable'
         or 'empowered' each agent is. For instance:
           * 1.0 means full capacity (the agent is not restricted).
           * 0.0 effectively disables the agent (it cannot act).
           * Intermediate values partially limit or shape the agent’s behavior.

    The typical workflow is:
      1) Construct GPTAPI with a SageMaker predictor (e.g., from sagemaker).
      2) Call high-level methods (select_action, set_control_value, etc.)
      3) The server (serve.py) receives these actions, processes them, and
         returns JSON, which GPTAPI then converts into NumPy arrays where needed.

    This class ensures that all method inputs are converted to JSON-friendly
    nested lists (or scalars) before sending, and that responses are converted
    back to NumPy arrays or Python data structures where appropriate.
    """

    def __init__(self, predictor):
        """
        :param predictor: An object (typically a sagemaker Predictor) that has 
                          `.predict(bytes)` and `.endpoint_name` attributes.
        """
        self.__predictor = predictor
        self.endpoint_name = self.__predictor.endpoint_name

    def _invoke(self, action: str, args: dict) -> Dict:
        """
        Internal helper for sending JSON payloads to the SageMaker endpoint 
        and parsing JSON responses.

        :param action: String name of the server-recognized action.
        :param args: A dict of arguments to include in the request payload.
        :return: Parsed JSON response (dict).
        """
        payload = {"action": action, "args": args}
        request_str = json.dumps(payload)
        response_bytes = self.__predictor.predict(request_str.encode("utf-8"))
        response_str = response_bytes.decode("utf-8")
        return json.loads(response_str)

    # -------------------------------------------------------------------------
    # Core RL/GPT actions (mirroring serve.py endpoints)
    # -------------------------------------------------------------------------

    def select_action(
        self,
        agent_ids: Union[List[str], np.ndarray],
        observations: Union[List[np.ndarray], np.ndarray],
        terminated_agent_ids: Optional[Union[List[str], np.ndarray]] = None
    ):
        """
        Ask the server to select an action for the given agents and their
        corresponding observations.

        :param agent_ids:
            A list or NumPy array of agent identifiers (strings).
        :param observations:
            Observations for these agents. Could be a list of arrays or a 
            single array. The server typically expects a batch dimension 
            matching agent_ids.
        :param terminated_agent_ids:
            Optional. If specified, a list/array of agent IDs that have just 
            terminated this step. The server may deregister them or handle 
            them internally.
        :return:
            A NumPy array of actions (np.float32), shape depends on the server’s
            action space. (The server typically returns {"action": ...}.)
        """
        agent_ids = convert_ndarrays_to_nested_lists(agent_ids)
        observations = convert_ndarrays_to_nested_lists(observations)
        terminated_agent_ids = convert_ndarrays_to_nested_lists(terminated_agent_ids)
        args = {
            "agent_ids": agent_ids,
            "observations": observations,
            "terminated_agent_ids": terminated_agent_ids,
        }
        response = self._invoke("select_action", args)
        action_data = response.get("action")
        action_data = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        return action_data

    def sample_observation(self, num_agents: int = None):
        """
        Request a sample observation from the server (e.g., from an 
        observation space distribution).

        :param num_agents:
            If provided, how many agents' observations to sample.
            If None, the server decides (often returns a single observation).
        :return:
            A NumPy array (np.float32) or nested array structure with the 
            sampled observations.
        """
        response = self._invoke("sample_observation", {"num_agents": num_agents})
        observation = response.get("observation", None)
        return convert_nested_lists_to_ndarrays(observation, dtype=np.float32)

    def sample_action(self, num_agents: int = None):
        """
        Request a sample action from the server (e.g., from an 
        action space distribution).

        :param num_agents:
            If provided, how many agents' actions to sample.
            If None, the server decides (often returns a single action).
        :return:
            A NumPy array (np.float32) or nested array structure with the 
            sampled actions.
        """
        response = self._invoke("sample_action", {"num_agents": num_agents})
        action = response.get("action", None)
        return convert_nested_lists_to_ndarrays(action, dtype=np.float32)

    def terminate_agents(self, terminated_agent_ids):
        """
        Deregister or terminate multiple agents on the server.

        :param terminated_agent_ids:
            A list/array of agent IDs (strings) to terminate.
        :return:
            The server response, which may include a message about which agents 
            were terminated successfully.
        """
        terminated_agent_ids = convert_ndarrays_to_nested_lists(terminated_agent_ids)
        response = self._invoke("terminate_agents", {"terminated_agent_ids": terminated_agent_ids})
        return response

    def reset_agents(self, max_agents: int = None):
        """
        Reset all agents in the environment, optionally specifying a new 
        maximum capacity.

        :param max_agents:
            If provided, sets a new maximum number of agents after reset. 
            If None, the existing max is retained.
        :return:
            The server’s reset response (often includes a status message).
        """
        args = {}
        if max_agents is not None:
            args["max_agents"] = int(max_agents)
        return self._invoke("reset_agents", args)

    def get_num_agents(self):
        """
        Retrieve the current number of active agents.

        :return:
            An integer for how many agents are currently active.
        """
        response = self._invoke("get_num_agents", {})
        num_agents = response.get("num_agents", 0)
        return int(num_agents)

    def get_agent_ids(self):
        """
        Get a list of all active agent IDs.

        :return:
            A NumPy array of strings representing agent IDs.
        """
        response = self._invoke("get_agent_ids", {})
        agent_ids = response.get("agent_ids", [])
        return convert_nested_lists_to_ndarrays(agent_ids, dtype=object)

    # -------------------------------------------------------------------------
    # GPT Input Sequence Length (num_input_states)
    # -------------------------------------------------------------------------

    def set_num_input_states(self, num_input_states: int):
        """
        Set how many past timesteps the GPT model uses as context. 
        Must be <= get_max_input_states().

        :param num_input_states: 
            An integer specifying the new context-window size for all agents.
        :return:
            Boolean indicating whether the update succeeded (True) or not (False).
        """
        response = self._invoke("set_num_input_states", {"num_input_states": num_input_states})
        return response.get("success", False)

    def get_num_input_states(self):
        """
        Get the current context-window size (how many past timesteps 
        are included).

        :return:
            An integer for the number of input states in use.
        """
        response = self._invoke("get_num_input_states", {})
        num_input_states = response.get("num_input_states", None)
        return int(num_input_states)

    def get_max_input_states(self):
        """
        Get the maximum allowable context-window size (num_input_states).

        :return:
            An integer for the max number of states the server can store.
        """
        response = self._invoke("get_max_input_states", {})
        max_input_states = response.get("max_input_states", None)
        return int(max_input_states)

    # -------------------------------------------------------------------------
    # Agent Performance Control Value
    # -------------------------------------------------------------------------

    def set_control_value(
        self,
        agent_ids: Union[List[str], np.ndarray] = None,
        control_value: Union[float, List[float], np.ndarray] = 1.0
    ):
        """
        Overwrite the control value(s) for specific agents or all agents. 
        Valid range is [0.0, 1.0].

        :param agent_ids:
            None to apply to all agents, or a list/array of agent IDs (strings).
        :param control_value:
            Either a single float in [0.0, 1.0] or a list/array of floats 
            in that range. If multiple floats, must match len(agent_ids).
        :return:
            Boolean: True on success, False otherwise. Check the server
            response's 'message' for more details if needed.
        """
        agent_ids = convert_ndarrays_to_nested_lists(agent_ids)
        control_value = convert_ndarrays_to_nested_lists(control_value)

        response = self._invoke(
            "set_control_value",
            {"agent_ids": agent_ids, "control_value": control_value}
        )
        return response.get("success", False)

    def get_control_value(self, agent_ids: Union[List[str], np.ndarray] = None):
        """
        Retrieve the current control values for specified agents (or all agents).

        :param agent_ids:
            None for all agents, or a list/array of agent IDs (strings).
        :return:
            A server response (often a dict mapping agent_id -> float).
        """
        agent_ids = convert_ndarrays_to_nested_lists(agent_ids)
        response = self._invoke("get_control_value", {"agent_ids": agent_ids})
        return response

    def status(self, agent_ids: Union[str, List[str], np.ndarray] = None):
        """
        Query the server’s overall status, optionally filtered by agent ID(s).

        :param agent_ids:
            None for all, a single agent ID string, or a list/array of IDs.
        :return:
            A dictionary representing the server’s status (keys/structure 
            depend on the server implementation).
        """
        agent_ids = convert_ndarrays_to_nested_lists(agent_ids)
        response = self._invoke("status", {"agent_ids": agent_ids})
        return response
