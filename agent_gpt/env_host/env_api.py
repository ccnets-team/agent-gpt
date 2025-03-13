from websocket._exceptions import WebSocketTimeoutException, WebSocketConnectionClosedException
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
               env_idx, num_agents):
        self.env_wrapper = env_wrapper
        self.environments = {}
        self.shutdown_event = threading.Event()
        self.ws = websocket.WebSocket()
        print("Connecting to Agent GPT server..., ", agent_gpt_server_url)
        self.ws.connect(agent_gpt_server_url)
        self.init_environment(remote_training_key, env_idx, num_agents)
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
                packed_request = self.ws.recv()
            except (socket.timeout, WebSocketTimeoutException):
                continue  # Silently continue without logging
            except WebSocketConnectionClosedException:
                logging.warning("WebSocket connection closed by server.")
                break
            except Exception as e:
                logging.exception("WebSocket receiving error: %s", e)
                continue
            try:
                # Unpack received request payload
                payload = self.unpack_request(packed_request)
                data = payload.get("data", {})
                method = data.get("method")
                env_key = data.get("env_key")

                # Execute method based on request
                if method == "make":
                    result = self.make(env_key, data.get("env_id"), data.get("render_mode"))
                elif method == "make_vec":
                    result = self.make_vec(env_key, data.get("env_id"), int(data.get("num_envs", 1)))
                elif method == "reset":
                    result = self.reset(env_key, data.get("seed"), data.get("options"))
                elif method == "step":
                    result = self.step(env_key, data.get("action"))
                elif method == "close":
                    result = self.close(env_key)
                elif method == "observation_space":
                    result = self.observation_space(env_key)
                elif method == "action_space":
                    result = self.action_space(env_key)
                else:
                    result = self.report_message(f"Unknown method: {method}")

                packed_response = self.pack_response(result)
                self.ws.send(packed_response)

            except Exception as e:
                logging.exception("Error processing message: %s", e)
                error_report = self.report_message(f"Internal server error: {str(e)}")
                self.ws.send(error_report)

        # if self.ws:
        #     print("Closing WebSocket connection.")
        #     self.ws.close()
        #     self.ws = None

    def pack_response(self, result):
        packed = msgpack.packb(result, use_bin_type=True)
        packed_response = base64.b64encode(packed).decode('utf-8')
        return packed_response

    def unpack_request(self, packed_request):
        packed_payload = base64.b64decode(packed_request)
        payload = msgpack.unpackb(packed_payload, raw=False)
        return payload
    
    def pack_request(self, request, training_key=None, data=None, message=None):
        payload = {"action": request}
        if training_key is not None:
            payload["training_key"] = training_key
        if message is not None:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
        return json.dumps(payload)

    def unpack_response(self, response):
        disclosed_message = json.loads(response)
        message = disclosed_message.get("message")
        data = disclosed_message.get("data")
        return data, message 
        
    def init_environment(self, remote_training_key: str, env_idx: int, num_agents: int):
        self.ws.send(json.dumps({
            "action": "init",
            "training_key": remote_training_key,
            "data": {  
                "env_idx": env_idx,
                "num_agents": num_agents
            }
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