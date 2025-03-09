import numpy as np
import logging
import websocket
import json
import socket
import queue
import threading
from typing import Optional, Any

# ------------------------------------------------
# Utility imports
# ------------------------------------------------
from ..utils.conversion_utils import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
    replace_nans_infs,
    space_to_dict,
)

URL = "wss://<agentgpt-api>.execute-api.<agentgpt-region>.amazonaws.com/<stage>/"
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500
WEBSOCKET_TIMEOUT = 10

class WSEnvAPI:
    def __init__(self, env_wrapper, host, port):
        self.env_wrapper = env_wrapper
        self.environments = {}
        self.host = host
        self.port = port
        self.ws = None
        self.client_connection_id = None
        self.message_queue = queue.Queue()

        self.shutdown_event = threading.Event()
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.ws:
            print("Closing WebSocket connection.", "connection ID:", self.client_connection_id)
            self.ws.close()
        for env_key in list(self.environments.keys()):
            self.environments[env_key].close()
            del self.environments[env_key]

    def run_server(self):
        self.communicate_forever()

    def communicate_forever(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(URL)
        self.ws.send(json.dumps({"action": "getId"}))
        self.client_connection_id = self.ws.recv()

        while not self.shutdown_event.is_set():
            try:
                message = self.ws.recv()
            except socket.timeout:
                continue  # Non-blocking via timeout, check shutdown_event periodically
            except websocket.WebSocketConnectionClosedException:
                logging.warning("WebSocket connection closed by server.")
                break
            except Exception as e:
                logging.exception("WebSocket receiving error: %s", e)
                continue

            try:
                sender_id, payload = self.disclose_message(message)
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

                payload = self.enclose_message(sender_id, response)
                self.ws.send(payload)

            except Exception as e:
                logging.exception("Error processing message: %s", e)
                error_payload = self.report_message(f"Internal server error: {str(e)}")
                self.ws.send(self.enclose_message(sender_id, error_payload))

        if self.ws:
            print("Closing WebSocket connection.", "connection ID:", self.client_connection_id)
            self.ws.close()
            self.ws = None

    def enclose_message(self, receive_id: str, payload: dict) -> bytes:
        encoded_data = json.dumps(payload).encode('utf-8')
        encoded_receive_id = receive_id.encode('utf-8')
        return encoded_receive_id + encoded_data

    def disclose_message(self, received_message):
        if isinstance(received_message, bytes):
            message = received_message.decode("utf-8")
        else:
            message = received_message
        sender_id = message[:16]
        payload = json.loads(message[16:])
        return sender_id, payload

    def report_message(self, message: str) -> str:
        return json.dumps({
            "action": "report",
            "status": HTTP_BAD_REQUEST,
            "message": message,
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
