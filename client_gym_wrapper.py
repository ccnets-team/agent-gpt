# client_gym_wrapper.py - A simple Flask API wrapper for OpenAI Gym environments.
from flask import Flask, request, jsonify
import numpy as np
import logging
from environments.factory import EnvironmentFactory
from environments.unity_backend import UnityBackend

EnvironmentFactory.register(UnityBackend)

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
        obs, _ = env.reset(seed=seed, options=options)
        
        return jsonify({"observation": obs.tolist() if isinstance(obs, np.ndarray) else [o.tolist() for o in obs]})

    def step(self):
        env_key = request.json.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), 400

        env = self.environments[env_key]["env"]
        is_vectorized = self.environments[env_key]["is_vectorized"]
                
        action = request.json.get("action")
        action = np.array(action, dtype=np.float32) if isinstance(action, list) else action

        if is_vectorized and action.ndim > 2:
            action = action[0]  # Adjust action dimensions for vectorized environments if needed

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        if isinstance(observation, np.ndarray):
            observation = observation.tolist()
        elif isinstance(observation, list):
            observation = [o.tolist() if isinstance(o, np.ndarray) else o for o in observation]
        else:
            observation = observation.tolist()
            
        reward = reward.tolist() if isinstance(reward, np.ndarray) else reward
        terminated = terminated.tolist() if isinstance(terminated, np.ndarray) else terminated
        truncated = truncated.tolist() if isinstance(truncated, np.ndarray) else truncated
                
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                info[key] = value.tolist()
            elif isinstance(value, list):
                info[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                assert not isinstance(value, np.ndarray), f"Unexpected ndarray in key: {key}"
                
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
        print("action_space type", action_space.__class__.__name__)

        action_space_type = action_space.__class__.__name__
        response = {"type": action_space_type}

        # Dynamically handle the attributes based on the action space type
        if action_space_type == "Discrete":
            response.update({
                "n": getattr(action_space, "n", None),
            })
        elif action_space_type == "MultiDiscrete":
            response.update({
                "shape": getattr(action_space, "shape", None),
                "nvec": getattr(action_space, "nvec", None).tolist() if hasattr(action_space, "nvec") else None,
            })
        elif action_space_type == "Box":
            response.update({
                "shape": getattr(action_space, "shape", None),
                "dtype": str(getattr(action_space, "dtype", None)),
                "low": getattr(action_space, "low", None).tolist() if hasattr(action_space, "low") else None,
                "high": getattr(action_space, "high", None).tolist() if hasattr(action_space, "high") else None,
            })
        else:
            response.update({
                "error": f"Unsupported action space type: {action_space_type}"
            })
        return jsonify(response)

    def observation_space(self):
        env_key = request.args.get("env_key")
        if env_key not in self.environments:
            return jsonify({"error": "Environment not initialized. Please call /make first."}), 400

        observation_space = self.environments[env_key]["env"].observation_space
        print("observation_space type", observation_space.__class__.__name__)
        return jsonify({
            "dtype": str(observation_space.dtype) if hasattr(observation_space, 'dtype') else None,
            "shape": list(observation_space.shape),
            "high": observation_space.high.tolist() if hasattr(observation_space, 'high') else None,
            "low": observation_space.low.tolist() if hasattr(observation_space, 'low') else None,
            "type": observation_space.__class__.__name__
        })
        
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
