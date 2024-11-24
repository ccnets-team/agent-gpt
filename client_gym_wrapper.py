# client_gym_wrapper.py - A simple Flask API wrapper for OpenAI Gym environments.
from flask import Flask, request, jsonify
import numpy as np
import logging
from environments.factory import EnvironmentFactory
from environments.unity_backend import UnityBackend
from environments.mujoco_backend import MujocoBackend
from utils.gym_space import serialize_space

EnvironmentFactory.register(UnityBackend)

def convert_list_to_ndarray(data, dtype):
    """
    Recursively convert all lists in the given data structure (dict, list, tuple)
    to numpy arrays, preserving the original structure and handling None values.

    Args:
        data: The input data, which can be a dict, list, tuple, or any other type.

    Returns:
        The data with all lists converted to numpy arrays where applicable.
    """
    if isinstance(data, list):
        # Convert the list to an ndarray if it contains no None values
        if all(item is not None for item in data):
            return np.array([convert_list_to_ndarray(item, dtype) for item in data], dtype=dtype)
        else:
            # Keep as list if there are None values
            return [convert_list_to_ndarray(item, dtype) if item is not None else None for item in data]
    elif isinstance(data, tuple):
        # Process tuple elements individually, preserving tuple structure
        return tuple(convert_list_to_ndarray(item, dtype) for item in data)
    elif isinstance(data, dict):
        # Process dict values recursively
        return {key: convert_list_to_ndarray(value, dtype) for key, value in data.items()}
    else:
        # Return the element as is if it's not a list, tuple, or dict
        return data
    
def convert_ndarray_to_list(data):
    """
    Recursively convert all numpy arrays in the given data structure (dict, list, tuple)
    to lists, preserving the original structure.

    Args:
        data: The input data, which can be a dict, list, tuple, or np.ndarray.

    Returns:
        The data with all numpy arrays converted to lists.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_ndarray_to_list(item) for item in data)
    elif isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    else:
        return data  # Return the element as is if it's not an ndarray, list, tuple, or dict

class ClientGymWrapper:
    def __init__(self):
        self.app = Flask(__name__)
        self.environments = {}
        self.env_cnt = 0    
        logging.basicConfig(level=logging.INFO)

        # Define routes
        self._define_routes()

    def _define_routes(self):   
        self.app.add_url_rule("/make", "make", self.make, methods=["POST"])
        self.app.add_url_rule("/make_vec", "make_vec", self.make_vec, methods=["POST"])
        self.app.add_url_rule("/reset", "reset", self.reset, methods=["POST"])
        self.app.add_url_rule("/step", "step", self.step, methods=["POST"])
        self.app.add_url_rule("/action_space", "action_space", self.action_space, methods=["GET"])
        self.app.add_url_rule("/observation_space", "observation_space", self.observation_space, methods=["GET"])
        self.app.add_url_rule("/close", "close", self.close, methods=["POST"])
        
    def make(self):
        env_id = request.json.get("env_id", "Humanoid-v4")  # Default to "Humanoid-v4" if not provided
        env_key = request.json.get("env_key", None)  # Generate a unique key if not provided        
        
        # Store environment and metadata    
        self.environments[env_key] = {
            # "env": EnvironmentFactory.make(env_id),
            "env": EnvironmentFactory.make(env_id, seed = self.env_cnt),
            "is_vectorized": False
        }
        self.env_cnt += 1
        logging.info(f"Environment {env_id} created with key {env_key}.")
        return jsonify({"message": f"Environment {env_id} created.", "env_key": env_key})

    def make_vec(self):
        env_id = request.json.get("env_id", "Humanoid-v4")  # Default to "Humanoid-v4" if not provided
        num_envs = request.json.get("num_envs", 1)  # Optional parameter for vectorized environments
        env_key = request.json.get("env_key", None)  # Generate a unique key if not provided        

        # Store vectorized environment and metadata
        self.environments[env_key] = {
            # "env": EnvironmentFactory.make_vec(env_id, num_envs=num_envs),
            "env": EnvironmentFactory.make_vec(env_id, num_envs=num_envs, seed = self.env_cnt),
            "is_vectorized": True
        }
        self.env_cnt += num_envs
        logging.info(f"Vectorized environment {env_id} created with {num_envs} instances, key {env_key}.")
        return jsonify({"message": f"Environment {env_id} created with {num_envs} instance(s).", "env_key": env_key})

    def reset(self):
        env_key = request.json.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), 400

        seed = request.json.get("seed", None)
        options = request.json.get("options", None)
        env = self.environments[env_key]["env"]
        observation, info = env.reset(seed=seed, options=options)
        
        observation = convert_ndarray_to_list(observation)
        info = convert_ndarray_to_list(info)
                    
        return jsonify({"observation": observation, "info": info})

    def step(self):
        env_key = request.json.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), 400

        env = self.environments[env_key]["env"]
                
        action = request.json.get("action")
        # Check if action is a list of lists (for Tuple or MultiDiscrete action spaces)
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        elif isinstance(action, tuple):
            action = tuple(np.array(a, dtype=np.float32) for a in action)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        observation = convert_ndarray_to_list(observation)
        reward = convert_ndarray_to_list(reward)
        terminated = convert_ndarray_to_list(terminated)
        truncated = convert_ndarray_to_list(truncated)
        info = convert_ndarray_to_list(info)
               
        # Serialize observations, rewards, terminations, and truncations
        response = {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        return jsonify(response)

    def action_space(self):
        env_key = request.args.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), 400

        action_space = self.environments[env_key]["env"].action_space
        action_space_info = serialize_space(action_space)

        return jsonify(action_space_info)

    def observation_space(self):
        env_key = request.args.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), 400

        observation_space = self.environments[env_key]["env"].observation_space
        observation_space_info = serialize_space(observation_space)

        return jsonify(observation_space_info)
        
    def close(self):
        env_key = request.json.get("env_key")
        if env_key in self.environments:
            self.environments[env_key]["env"].close()
            del self.environments[env_key]
            logging.info(f"Environment with key {env_key} closed.")
            return jsonify({"message": f"Environment with key {env_key} closed successfully."})
        return jsonify({"error": "No environment with this key to close."}), 400

    def run(self, port):
        logging.info(f"Starting Gym API server on port {port}.")
        self.app.run(port=port)

if __name__ == "__main__":
    server = ClientGymWrapper()
    server.run(port=5000)
