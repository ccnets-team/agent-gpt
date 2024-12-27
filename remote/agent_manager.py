# agent_gpt.py
import logging
import requests
import numpy as np

from utils.data_converters import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays
)

HTTP_OK = 200

class AgentManager:
    """
    Client-side class for interacting with a remote 'EnvRemoteRunner' service.
    Responsible for sending HTTP requests that perform model loading,
    agent registration, action selection, etc.
    """

    def __init__(self, api_url: str, model_path: str):
        """
        :param api_url: Base URL of the remote service (e.g. http://localhost:5000).
        :param model_path: Location of the model (e.g. s3://...).
        """
        self.api_url = api_url
        self.model_path = model_path

        logging.basicConfig(level=logging.INFO)

    # -------------------------------------------------------------------------
    # Internal HTTP helpers
    # -------------------------------------------------------------------------
    def _post(self, endpoint: str, payload: dict = None):
        """
        Helper to send a POST request with optional JSON payload.
        Throws a RuntimeError if the response status is not HTTP_OK.
        Returns the decoded JSON response.
        """
        payload = payload or {}
        resp = requests.post(f"{self.api_url}/{endpoint}", json=payload)
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"POST /{endpoint} failed: {resp.text}")

        data = resp.json()
        logging.info(f"[POST /{endpoint}] response: {data}")
        return data

    def _get(self, endpoint: str):
        """
        Helper to send a GET request.
        Throws a RuntimeError if the response status is not HTTP_OK.
        Returns the decoded JSON response.
        """
        resp = requests.get(f"{self.api_url}/{endpoint}")
        if resp.status_code != HTTP_OK:
            raise RuntimeError(f"GET /{endpoint} failed: {resp.text}")

        data = resp.json()
        logging.info(f"[GET /{endpoint}] response: {data}")
        return data

    # -------------------------------------------------------------------------
    # Model management
    # -------------------------------------------------------------------------
    def load_model(self, model_path: str):
        """
        Loads a model on the remote runner (POST /load_model).
        """
        return self._post("load_model", {"model_path": model_path})

    # -------------------------------------------------------------------------
    # Performance controls
    # -------------------------------------------------------------------------
    def control_performance(self, performance: float):
        """
        Sets a global performance level on the remote runner (POST /control_performance).
        """
        self._post("control_performance", {"performance": performance})

    def control_agent_performance(self, agent_ids: list, performances: list):
        """
        Sets per-agent performance levels (POST /control_agent_performance).
        """
        self._post("control_agent_performance", {
            "agent_ids": agent_ids,
            "performances": performances
        })

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------
    def select_action(self, agent_id: int, observation):
        """
        Convenience method for single agent + single observation.
        """
        actions = self.select_actions([agent_id], [observation])
        return actions[0] if actions is not None else None

    def select_actions(self, agent_ids: list, observations: list):
        """
        Requests actions for multiple agents (POST /select_action).
        """
        obs_converted = convert_ndarrays_to_nested_lists(observations)
        data = self._post("select_action", {
            "agent_id": agent_ids,
            "observation": obs_converted
        })
        # Convert action data back to np.ndarray
        action_data = data.get("action")
        if action_data is not None:
            action_data = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        return action_data

    # -------------------------------------------------------------------------
    # GPT-specific endpoints
    # -------------------------------------------------------------------------
    def get_gpt_seq_len(self) -> int:
        """
        Retrieves the GPT sequence length (GET /get_gpt_seq_len).
        """
        data = self._get("get_gpt_seq_len")
        return data.get("seq_len", 0)

    def get_max_seq_len(self) -> int:
        """
        Retrieves the GPT sequence length (GET /get_max_seq_len).
        """
        data = self._get("get_max_seq_len")
        return data.get("seq_len", 0)    

    def set_input_seq_len(self, seq_len: int) -> bool:
        """
        Sets an input sequence length (POST /set_input_seq_len).
        Returns True if successful.
        """
        data = self._post("set_input_seq_len", {"seq_len": seq_len})
        return data.get("success", False)

    # -------------------------------------------------------------------------
    # Agent management
    # -------------------------------------------------------------------------
    def register_agents(self, agent_ids: list = None):
        """
        Registers multiple agents (POST /register_agents).
        """
        agent_ids = agent_ids or []
        return self._post("register_agents", {"agent_ids": agent_ids})

    def delete_agents(self, agent_ids: list):
        """
        Deregisters multiple agents (POST /deregister_agent).
        Returns how many agents remain, if the server provides it.
        """
        data = self._post("deregister_agent", {"agent_ids": agent_ids})
        return data.get("num_left_agents")

    def reset_agents(self):
        """
        Resets agent(s) to initial states (POST /reset_agents).
        """
        return self._post("reset_agents", {})

    def get_registered_agents(self) -> list:
        """
        Returns the list of registered agents (GET /get_registered_agents).
        """
        data = self._get("get_registered_agents")
        return data.get("agent_ids", [])

    # -------------------------------------------------------------------------
    # Status & metrics
    # -------------------------------------------------------------------------
    def status(self):
        """
        Retrieves the overall system status (GET /status).
        """
        return self._get("status")

    def performance_status(self):
        """
        Retrieves the overall performance status (GET /performance_status).
        """
        return self._get("performance_status")

    def agent_performance_status(self):
        """
        Retrieves performance status per agent (GET /agent_performance_status).
        """
        return self._get("agent_performance_status")

    def agent_activity_status(self):
        """
        Retrieves the activity status of agents (GET /agent_activity_status).
        """
        return self._get("agent_activity_status")

