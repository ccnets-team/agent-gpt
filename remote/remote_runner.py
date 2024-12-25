import requests
import logging
import uuid
import numpy as np
from utils.data_converters import convert_ndarrays_to_nested_lists, convert_nested_lists_to_ndarrays
from .one_click_sagemaker import one_click_sagemaker_inference

HTTP_BAD_REQUEST = 400
HTTP_OK = 200

class RemoteRunner:
    env_simulator = None  # Class-level variable to store the backend
    
    """
    A client class for interacting with a remote EnvRemoteRunner server.
    It sends HTTP requests to endpoints like /make_env, /close_env, etc.
    and handles any necessary data conversions (NumPy <-> JSON).
    
    The remote server is expected to support:
      - POST /make_env
      - POST /close_env
      - POST /load_model
      - POST /register_agent
      - POST /deregister_agent
      - POST /select_action
      - POST /reset_agent
      - GET  /status
    """

    def __init__(self, api_url: str):
        """
        :param api_url: The base URL of the remote EnvRemoteRunner server
                        e.g. "http://localhost:5001"
        """
        self.api_url = api_url
        logging.basicConfig(level=logging.INFO)

    # -------------------------------------------------------------------------
    # Environment Management
    # -------------------------------------------------------------------------

    def make_env(self, env_id="Humanoid-v4", env_key=None, num_envs=1):
        """
        Create (or connect to) a new environment on the remote runner.
        This calls POST /make_env on the server.

        :param env_id: The environment ID (e.g., "Humanoid-v4").
        :param env_key: Optional custom key to identify this environment.
        :param num_envs: Number of parallel/vectorized envs if supported.
        :return: JSON response with fields like {"message", "env_key", "num_envs"}.
        """
        if not env_key:
            env_key = str(uuid.uuid4())

        payload = {
            "env_id": env_id,
            "env_key": env_key,
            "num_envs": num_envs
        }
        response = requests.post(f"{self.api_url}/make_env", json=payload)
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to make_env: {response.text}")

        data = response.json()
        logging.info(f"make_env response: {data}")
        return data

    def close_env(self, env_key: str):
        """
        Close the specified environment.
        Calls POST /close_env on the server.

        :param env_key: The unique key identifying the environment to close.
        :return: JSON response, e.g. {"message": "..."}
        """
        payload = {"env_key": env_key}
        response = requests.post(f"{self.api_url}/close_env", json=payload)
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to close_env: {response.text}")

        data = response.json()
        logging.info(f"close_env response: {data}")
        return data

    # -------------------------------------------------------------------------
    # Model & Agent Management + Action Selection
    # -------------------------------------------------------------------------

    def load_model(self, model_path: str):
        """
        Load the model on the remote runner.
        Calls POST /load_model.

        :param model_path: Path to your model (e.g., "s3://bucket/model.onnx").
        :return: JSON response
        """
        payload = {"model_path": model_path}
        response = requests.post(f"{self.api_url}/load_model", json=payload)
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to load_model: {response.text}")

        data = response.json()
        logging.info(f"load_model response: {data}")
        return data

    def select_action(self, agent_id: list, observation: list):
        """
        Calls POST /select_action to get an action for the given agent.

        :param agent_id: The agent's unique ID.
        :param observation: The current observation (NumPy array or list).
        :return: (action, info) from the server, converted to local NumPy if needed.
        """
        observation = convert_ndarrays_to_nested_lists(observation)

        payload = {
            "agent_id": agent_id,
            "observation": observation
        }
        response = requests.post(f"{self.api_url}/select_action", json=payload)
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to select_action: {response.text}")

        data = response.json()
        action = data.get("action")
        if action is not None:
            action = convert_nested_lists_to_ndarrays(action, dtype=np.float32)

        logging.info(f"select_action response: {data}")
        return action

    def delete_agents(self, agent_ids: list):
        """
        Calls POST /deregister_agent to remove the specified agent from the system.

        :param agent_id: The agent's unique ID.
        :return: JSON response
        """
        payload = {"agent_ids": agent_ids}
        response = requests.post(f"{self.api_url}/deregister_agent", json=payload)
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to deregister_agent: {response.text}")

        data = response.json()
        logging.info(f"deregister_agent response: {data}")
        return data

    def reset_agent(self):
        """
        Calls POST /reset_agent to reset the agent's internal state.

        :param agent_id: The agent's unique ID.
        :return: JSON response
        """
        response = requests.post(f"{self.api_url}/reset_agent")
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to reset_agent: {response.text}")

        data = response.json()
        logging.info(f"reset_agent response: {data}")
        return data

    def status(self):
        """
        Calls GET /status to retrieve system information.
        :return: JSON object with fields like { "model_loaded", "num_agents", "details" }
        """
        response = requests.get(f"{self.api_url}/status")
        if response.status_code != HTTP_OK:
            raise RuntimeError(f"Failed to get status: {response.text}")

        data = response.json()
        logging.info(f"status response: {data}")
        return data

    @classmethod
    def run(cls, env_simulator, port=5000):
        """
        Register a backend and start the EnvironmentGateway server.
        :param backend: Backend class implementing `make` and `make_vec` methods.
        :param port: Port to run the server on.
        """
        cls.env_simulator = env_simulator
        logging.info(f"EnvSimulator {env_simulator.__name__} registered successfully.")

        logging.info(f"Starting EnvironmentRunway server on port {port} with backend {env_simulator.__name__}.")
        one_click_sagemaker_inference() 
        
        return cls
