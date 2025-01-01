# env_host.py
from flask import request, jsonify
import numpy as np
import logging
from utils.data_converters import convert_ndarrays_to_nested_lists, convert_nested_lists_to_ndarrays
from utils.gym_space import space_to_dict

HTTP_BAD_REQUEST = 400
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500

class EnvHost:
    def __init__(self, app, env_simulator):
        self.env_simulator = env_simulator
        self.environments = {}
        self.app = app 
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
        env_id = request.json.get("env_id", "Humanoid-v5")  # Default to "Humanoid-v4" if not provided
        if env_id is None and env_id.endswith("-aws"):
            env_id = env_id.rfind("-aws")
        
        env_key = request.json.get("env_key", None)  # Generate a unique key if not provided        

        if self.env_simulator is None or not hasattr(self.env_simulator, "make"):
            return jsonify({"error": "Backend not properly registered."}), HTTP_BAD_REQUEST
        
        # Store environment and metadata    
        self.environments[env_key] = {
            "env": self.env_simulator.make(env_id),
            "is_vectorized": False
        }
        logging.info(f"Environment {env_id} created with key {env_key}.")
        return jsonify({"message": f"Environment {env_id} created.", "env_key": env_key})

    def make_vec(self):
        env_id = request.json.get("env_id", "Humanoid-v5")  # Default to "Humanoid-v4" if not provided
        if env_id is None and env_id.endswith("-aws"):
            env_id = env_id.rfind("-aws")
                    
        num_envs = request.json.get("num_envs", 1)  # Optional parameter for vectorized environments
        env_key = request.json.get("env_key", None)  # Generate a unique key if not provided        

        if self.env_simulator is None or not hasattr(self.env_simulator, "make_vec"):
            return jsonify({"error": "Backend not properly registered."}), HTTP_BAD_REQUEST

        # Store vectorized environment and metadata
        self.environments[env_key] = {
            "env": self.env_simulator.make_vec(env_id, num_envs=num_envs),
            "is_vectorized": True
        }
        logging.info(f"Vectorized environment {env_id} created with {num_envs} instances, key {env_key}.")
        return jsonify({"message": f"Environment {env_id} created with {num_envs} instance(s).", "env_key": env_key})

    def reset(self):
        env_key = request.json.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), HTTP_BAD_REQUEST

        seed = request.json.get("seed", None)
        options = request.json.get("options", None)
        env = self.environments[env_key]["env"]
        observation, info = env.reset(seed=seed, options=options)
        
        observation, info = (
            convert_ndarrays_to_nested_lists(x) for x in (observation, info)
        )
                            
        return jsonify({"observation": observation, "info": info})

    def step(self):
        env_key = request.json.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), HTTP_BAD_REQUEST

        env = self.environments[env_key]["env"]
                
        action = request.json.get("action")
        action = convert_nested_lists_to_ndarrays(action, dtype=np.float32)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        observation, reward, terminated, truncated, info = (
            convert_ndarrays_to_nested_lists(x) for x in (observation, reward, terminated, truncated, info)
        )
               
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
            return jsonify({"error": "Environment not initialized. Please call /make first."}), HTTP_BAD_REQUEST

        action_space = self.environments[env_key]["env"].action_space
        action_space_serial = space_to_dict(action_space)

        return jsonify(action_space_serial)

    def observation_space(self):
        env_key = request.args.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), HTTP_BAD_REQUEST

        observation_space = self.environments[env_key]["env"].observation_space
        observation_space_serial = space_to_dict(observation_space)

        return jsonify(observation_space_serial)
        
    def close(self):
        if len(self.environments) == 0:
            return jsonify({"message": "No environment to close."})
        
        env_key = request.json.get("env_key", None)
        if env_key in self.environments:
            self.environments[env_key]["env"].close()
            del self.environments[env_key]
            logging.info(f"Environment with key {env_key} closed.")
            return jsonify({"message": f"Environment with key {env_key} closed successfully."})
        
        return jsonify({"error": "No environment with this key to close."}), HTTP_BAD_REQUEST
