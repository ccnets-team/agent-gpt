# env_host/env_api_websocket.py
import asyncio
import numpy as np
import logging
import uvicorn
from fastapi import WebSocket, WebSocketDisconnect, FastAPI
import websockets  # async websocket library
from typing import Optional, Any
import json
# ------------------------------------------------
# Utility imports
# ------------------------------------------------
from ..utils.conversion_utils import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
    replace_nans_infs,
    space_to_dict,
)

URL = "wss://<agent-gpt-api>.execute-api.us-east-1.amazonaws.com/"
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500

class WSEnvAPI:
    """
    EnvAPIWebSocket is a client that connects to a remote WebSocket endpoint
    (e.g., "wss://<agent-gpt-api>.execute-api.us-east-1.amazonaws.com/") and listens
    for gym methods to interact with an underlying RL environment.
    
    The supported commands include:
      - "make"
      - "make_vec"
      - "reset"
      - "step"
      - "close"
      - "get_action_space"
      - "get_observation_space"
      
    For each command, the corresponding method is invoked and the response is sent back
    over the WebSocket connection.
    """
    def __init__(self, env_wrapper,  host, port):
        self.env_wrapper = env_wrapper
        self.environments = {}
        self.host = host
        self.port = port
        self.session = None 
        self.app = FastAPI()
        self.clients = set()  # Set to hold connected local clients

        # Register the WebSocket route with the FastAPI instance.
        self.app.add_api_websocket_route("/ws", self.websocket_endpoint)
        
        # Register the startup event to launch the external WS listener as a background task.
        self.app.add_event_handler("startup", self.startup_event)

    def __exit__(self, exc_type, exc_value, traceback):
        for env_key in list(self.environments.keys()):
            self.environments[env_key].close()
            del self.environments[env_key]

    def run_server(self):
        """Run the FastAPI/Starlette application via uvicorn."""
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    async def startup_event(self):
        """Run on startup: launch the external WebSocket listener as a background task."""
        asyncio.create_task(self.connect_and_listen())

    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        print("Client connected to /ws")
        try:
            while True:
                message = await websocket.receive()
                # FastAPI's receive() returns a dict with keys "text" or "bytes"
                if "text" in message:
                    print(f"Received text from client: {message['text']}")
                elif "bytes" in message:
                    try:
                        decoded = json.loads(message["bytes"].decode("utf-8"))
                        print(f"Received binary from client: {decoded}")
                    except Exception as e:
                        print(f"Error decoding binary message: {e}")
        except WebSocketDisconnect:
            print("Client disconnected from /ws endpoint")
            self.clients.remove(websocket)

    async def connect_and_listen(self):
        async with websockets.connect(URL) as ws:
            async for message in ws:
                # Check if message is binary (bytes) or already text
                if isinstance(message, bytes):
                    try:
                        data = json.loads(message.decode("utf-8"))
                    except Exception as e:
                        print(f"Invalid binary JSON from external WS: {message}")
                        continue
                else:
                    try:
                        data = json.loads(message)
                    except Exception as e:
                        print(f"Invalid JSON from external WS: {message}")
                        continue
                if "method" not in data:
                    print(f"Missing 'method' key in message: {data}")
                    continue
                
                method = data.get("method")
                if method == "make":
                    response = await asyncio.to_thread(
                        self.make, data.get("env_key"), data.get("env_id"), data.get("render_mode")
                    )
                elif method == "make_vec":
                    response = await asyncio.to_thread(
                        self.make_vec, data.get("env_key"), data.get("env_id"), int(data.get("num_envs"))
                    )
                elif method == "reset":
                    response = await asyncio.to_thread(
                        self.reset, data.get("env_key"), data.get("seed"), data.get("options")
                    )
                elif method == "step":
                    response = await asyncio.to_thread(
                        self.step, data.get("env_key"), data.get("action")
                    )
                elif method == "close":
                    response = await asyncio.to_thread(
                        self.close, data.get("env_key")
                    )
                elif method == "get_action_space":
                    response = await asyncio.to_thread(
                        self.action_space, data.get("env_key")
                    )
                elif method == "get_observation_space":
                    response = await asyncio.to_thread(
                        self.observation_space, data.get("env_key")
                    )
                else:
                    response = {"error": f"Unknown method: {method}"}
                    raise ValueError(f"Unknown method: {response}")

                # Encode the response as binary (JSON dumped then UTF-8 encoded)
                response_binary = json.dumps(response).encode("utf-8")
                for client in list(self.clients):
                    try:
                        await client.send_bytes(response_binary)
                    except Exception as e:
                        print(f"Error sending to client: {e}")
                        self.clients.remove(client)
            
    def attempt_register_env(self, env_id: str, env_entry_point: str, env_dir: str):
        """Register the environment if the necessary parameters are provided."""
        if env_id and (env_entry_point or env_dir):
            logging.info(f"Registering environment {env_id} with entry_point {env_entry_point}")
            self.env_wrapper.register(env_id=env_id, env_entry_point=env_entry_point, env_dir=env_dir)

    # ------------------------------------------------
    # The methods each command calls (returning Python dicts)
    # ------------------------------------------------
    def make(self, env_key: str, env_id: str, render_mode: Optional[str] = None):
        if not self.env_wrapper or not hasattr(self.env_wrapper, "make"):
            return {"error": "Backend not properly registered."}
        env_instance = self.env_wrapper.make(env_id, render_mode=render_mode)
        self.environments[env_key] = env_instance
        logging.info(f"Environment {env_id} created with key {env_key}.")
        return {"message": f"Environment {env_id} created.", "env_key": env_key}

    def make_vec(self, env_key: str, env_id: str, num_envs: int):
        if not self.env_wrapper or not hasattr(self.env_wrapper, "make_vec"):
            return {"error": "Backend not properly registered."}
        env_instance = self.env_wrapper.make_vec(env_id, num_envs=num_envs)
        self.environments[env_key] = env_instance
        logging.info(f"Vectorized env {env_id} with {num_envs} instance(s), key {env_key}.")
        return {"message": f"Environment {env_id} created with {num_envs} instance(s).", "env_key": env_key}

    def reset(self, env_key: str, seed: Optional[int], options: Optional[Any]):
        if env_key not in self.environments:
            return {"error": "Environment not initialized. Please call make first."}
        env = self.environments[env_key]
        observation, info = env.reset(seed=seed, options=options)
        observation, info = tuple(convert_ndarrays_to_nested_lists(x) for x in (observation, info))
        return {"observation": observation, "info": info}

    def step(self, env_key: str, action_data):
        if env_key not in self.environments:
            return {"error": "Environment not initialized. Please call make first."}
        env = self.environments[env_key]
        action = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            logging.exception("Error in env.step()")
            return {"error": str(e)}
        observation, reward, terminated, truncated, info = tuple(
            convert_ndarrays_to_nested_lists(x)
            for x in (observation, reward, terminated, truncated, info)
        )
        return {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }

    def action_space(self, env_key: str):
        if env_key not in self.environments:
            return {"error": "Environment not initialized. Please call make first."}
        action_space = self.environments[env_key].action_space
        action_space = space_to_dict(action_space)
        return replace_nans_infs(action_space)

    def observation_space(self, env_key: str):
        if env_key not in self.environments:
            return {"error": "Environment not initialized. Please call make first."}
        observation_space = self.environments[env_key].observation_space
        observation_space = space_to_dict(observation_space)
        return replace_nans_infs(observation_space)

    def close(self, env_key: str):
        if env_key not in self.environments:
            # If the given key is not found, close all environments.
            for key in list(self.environments.keys()):
                self.environments[key].close()
                logging.info(f"Environment with key {key} closed.")
                del self.environments[key]
            return {"message": "All environments closed successfully."}
        self.environments[env_key].close()
        del self.environments[env_key]
        logging.info(f"Environment with key {env_key} closed.")
        return {"message": f"Environment {env_key} closed successfully."}