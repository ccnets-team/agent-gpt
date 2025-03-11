import numpy as np
import logging
import websocket
import json
import socket
import queue
import threading
from typing import Optional, Any
import msgpack
import base64
from websocket._exceptions import WebSocketTimeoutException, WebSocketConnectionClosedException

# ------------------------------------------------
# Utility imports
# ------------------------------------------------
from ..utils.conversion_utils import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
    replace_nans_infs,
    space_to_dict,
)

WEBSOCKET_TIMEOUT = 1
class EnvAPI:
    def __init__(self, env_wrapper, remote_training_key, agent_gpt_server_url, 
               env_id, num_envs, env_idx, num_agents, entry_point=None, env_dir=None, seed=None):
        self.env_wrapper = env_wrapper
        self.environments = {}
        self.message_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.ws = websocket.WebSocket()
        self.ws.connect(agent_gpt_server_url)        
        self.register_environment(self.ws, remote_training_key, 
                                    env_id, num_envs, env_idx, num_agents, env_dir, entry_point, seed)
        self.ws.settimeout(WEBSOCKET_TIMEOUT)
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.ws:
            print("Closing WebSocket connection.")
            self.ws.close()
        for env_key in list(self.environments.keys()):
            self.environments[env_key].close()  
            del self.environments[env_key]
        
    def communicate(self):
        while not self.shutdown_event.is_set():
            try:
                message = self.ws.recv()
            except (socket.timeout, WebSocketTimeoutException):
                continue  # Silently continue without logging
            except WebSocketConnectionClosedException:
                logging.warning("WebSocket connection closed by server.")
                break
            except Exception as e:
                logging.exception("WebSocket receiving error: %s", e)
                continue
            
            try:
                payload = self.disclose_message(message)
                data = payload.get("data", {})
                method = data.get("method")
                env_key = data.get("env_key")

                if method == "make":
                    response = self.make(env_key, data.get("env_id"), data.get("render_mode"))
                elif method == "make_vec":
                    response = self.make_vec(env_key, data.get("env_id"), int(data.get("num_envs", 1)))
                elif method == "reset":
                    response = self.reset(env_key, data.get("seed"), data.get("options"))
                elif method == "step":
                    response = self.step(env_key, data.get("action"))
                elif method == "close":
                    response = self.close(env_key)
                elif method == "observation_space":
                    response = self.observation_space(env_key)
                elif method == "action_space":
                    response = self.action_space(env_key)
                else:
                    response = self.report_message(f"Unknown method: {method}")

                message = self.enclose_message(response)
                self.ws.send(message)

            except Exception as e:
                logging.exception("Error processing message: %s", e)
                error_payload = self.report_message(f"Internal server error: {str(e)}")
                self.ws.send(self.enclose_message(error_payload))

        if self.ws:
            print("Closing WebSocket connection.")
            self.ws.close()
            self.ws = None

    def enclose_message(self, payload):
        packed = msgpack.packb(payload, use_bin_type=True)
        messagee = base64.b64encode(packed).decode('utf-8')
        return messagee

    def disclose_message(self, message):
        compressed = base64.b64decode(message)
        payload = msgpack.unpackb(compressed, raw=False)
        return payload
    
    # Example from client-side (your original snippet):
    def register_environment(ws: websocket.WebSocket, remote_training_key: str, 
                                env_id: str,  num_envs: int, env_idx: int, num_agents: int, env_dir: str, entry_point: str, seed: int):
        data = {
            "env_id": env_id,    
            "num_envs": num_envs,
            "env_idx": env_idx,
            "num_agents": num_agents,
            "env_dir": env_dir,
            "entry_point": entry_point,
            "seed": seed
        }
        ws.send(json.dumps({
            "action": "register",
            "training_key": remote_training_key,   
            "data": data
        }))

    def report_message(self, message: str, type: str = "error") -> str:
        return json.dumps({
            "action": "event",
            "message": message,
            "type": type
        })

    # ----------------- Environment methods -----------------

    def make(self, env_key: str, env_id: str, render_mode: Optional[str] = None):
        env_instance = self.env_wrapper.make(env_id, render_mode=render_mode)
        self.environments[env_key] = env_instance
        return {"message": f"Environment {env_id} created.", "env_key": env_key}

    def make_vec(self, env_key: str, env_id: str, num_envs: int):
        env_instance = self.env_wrapper.make_vec(env_id, num_envs=num_envs)
        self.environments[env_key] = env_instance
        return {"message": f"Vectorized environment {env_id} created.", "env_key": env_key}

    def reset(self, env_key: str, seed: Optional[int], options: Optional[Any]):
        env = self.environments[env_key]
        observation, info = env.reset(seed=seed, options=options)
        return {"observation": convert_ndarrays_to_nested_lists(observation), "info": convert_ndarrays_to_nested_lists(info)}

    def step(self, env_key: str, action_data):
        env = self.environments[env_key]
        action = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        return {
            "observation": convert_ndarrays_to_nested_lists(observation),
            "reward": convert_ndarrays_to_nested_lists(reward),
            "terminated": convert_ndarrays_to_nested_lists(terminated),
            "truncated": convert_ndarrays_to_nested_lists(truncated),
            "info": convert_ndarrays_to_nested_lists(info)
        }

    def action_space(self, env_key: str):
        return replace_nans_infs(space_to_dict(self.environments[env_key].action_space))

    def observation_space(self, env_key: str):
        return replace_nans_infs(space_to_dict(self.environments[env_key].observation_space))

    def close(self, env_key: str):
        if env_key in self.environments:
            self.environments[env_key].close()
            del self.environments[env_key]
            return {"message": f"Environment {env_key} closed."}
        return {"error": f"Environment {env_key} not found."}