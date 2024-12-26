#agent_gpt.py
import requests
import logging
import numpy as np
from utils.data_converters import convert_ndarrays_to_nested_lists, convert_nested_lists_to_ndarrays

HTTP_BAD_REQUEST = 400
HTTP_OK = 200

class RemoteAgentHost:
    def __init__(self,
                 model_path: str,
                 api_url: str = "http://oneclickrobotics-aws.ccnets.org"):
        """
        :param model_path: Where your model is stored (e.g. s3://...).
        :param api_url: The base URL of the remote service (EnvRemoteRunner).
        :param default_env_id: Default environment ID to use.
        """
        self.model_path = model_path
        self.api_url = api_url

        logging.basicConfig(level=logging.INFO)

    def load_model(self, model_path: str):
        """
        Loads a model on the remote runner (POST /load_model).
        """
        payload = {"model_path": model_path}
        resp = requests.post(f"{self.api_url}/load_model", json=payload)
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to load_model: {resp.text}")

        data = resp.json()
        logging.info(f"load_model response: {data}")
        return data
    
    def select_action(self, agent_ids: list, observations: list):
        """
        Calls POST /select_action to get an action for each agent.
        This is analogous to a ChatGPT prompt -> ChatGPT response cycle.
        """
        # Convert data so it can be JSON-serialized
        observations_converted = convert_ndarrays_to_nested_lists(observations)

        payload = {
            "agent_id": agent_ids,
            "observation": observations_converted
        }
        resp = requests.post(f"{self.api_url}/select_action", json=payload)
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to select_action: {resp.text}")

        data = resp.json()
        # e.g. data = {"action": [[0.1, 0.2, 0.3], ...]}
        action_data = data.get("action")
        if action_data is not None:
            action_data = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)

        logging.info(f"select_action response: {data}")
        return action_data

    def get_gpt_seq_len(self) -> int:
        """
        An example method that returns some 'sequence length' parameter.
        """
    
    def set_input_seq_len(self, seq_len: int) -> bool:
        """
        An example method that sets some 'sequence length' parameter
        and returns True for success (as shown in your usage snippet).
        """

    def set_model_performance(self, performance: float):
        """
        An example method that might tweak some performance-related parameter.
        """

    def register_agents(self, agent_ids: list = None):
        """
        Example for registering multiple agents at once (POST /register_agents).
        (In your original snippet, you called agent_gpt.register_agents().)
        """
        agent_ids = agent_ids or []
        payload = {
            "agent_ids": agent_ids
        }
        resp = requests.post(f"{self.api_url}/register_agents", json=payload)
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to register_agents: {resp.text}")

        data = resp.json()
        logging.info(f"register_agent response: {data}")
        return data

    def delete_agents(self, agent_ids: list):
        """
        Deregister agents (POST /deregister_agent).
        The response might indicate how many agents remain, etc.
        """
        payload = {"agent_ids": agent_ids}
        resp = requests.post(f"{self.api_url}/deregister_agent", json=payload)
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to deregister_agent: {resp.text}")

        data = resp.json()
        logging.info(f"deregister_agent response: {data}")
        # Suppose the server returns something like {"num_left_agents": <int>}
        return data.get("num_left_agents")

    def reset_agents(self):
        """
        Reset the agent's internal state (POST /reset_agents).
        Might accept a list of agent_ids if your server supports it.
        """
        payload = {[]}
        resp = requests.post(f"{self.api_url}/reset_agents", json=payload)
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to reset_agents: {resp.text}")

        data = resp.json()
        logging.info(f"reset_agents response: {data}")
        return data

    def status(self):
        """
        Retrieve system status (GET /status).
        """
        resp = requests.get(f"{self.api_url}/status")
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to get status: {resp.text}")

        data = resp.json()
        logging.info(f"status response: {data}")
        return data