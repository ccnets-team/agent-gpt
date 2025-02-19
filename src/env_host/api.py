# env_host/env_api.py
import numpy as np
import logging
import msgpack
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from typing import Optional, Any

# ------------------------------------------------
# Utility imports
# ------------------------------------------------
from src.utils.data_converters import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
    replace_nans_infs,
)
from src.utils.gym_space import space_to_dict

HTTP_BAD_REQUEST = 400
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500

# ------------------------------------------------
# EnvAPI class with FastAPI integration
# ------------------------------------------------
class EnvAPI:
    """
    EnvAPI is a minimal FastAPI service bridging an RL environment (e.g., Gym, Unity)
    with external trainers. It serves two primary roles:

      1) Receives inbound requests (e.g., /step, /reset) and delegates them
         to an underlying environment wrapper (your env simulator).
      2) Extends the basic Gymnasium protocol with improved security and remote
         communication methods for flexible, safer usage—whether hosted locally
         or in the cloud.
    """    
    def __init__(self, env_simulator, host: str = "0.0.0.0", port: int = 80):
        """
        env_simulator: an object that must have .make(...) and .make_vec(...)
        """
        self.env_simulator = env_simulator
        self.environments = {}
        self.host = host
        self.port = port

        # Create a FastAPI instance
        self.app = FastAPI()

        # Define all routes
        self._define_endpoints()
    
    def __exit__(self, exc_type, exc_value, traceback):
        for env_key in self.environments:
            self.environments[env_key].close()
            del self.environments[env_key]
    
    def run_server(self):
        """Run the FastAPI/Starlette application via uvicorn."""
        uvicorn.run(self.app, host=self.host, port=self.port, 
                    log_level="warning") # Only show warnings and errors
        
    def _define_endpoints(self):
        """Attach all routes/endpoints to self.app."""

        @self.app.post("/make")
        async def make_endpoint(request: Request) -> Response:
            """
            Equivalent to /make, but receives and returns msgpack.
            Expects:
              {
                "env_key": str,
                "env_id": str,
                "render_mode": Optional[str]
              }
            """
            raw_body = await request.body()
            try:
                body_data = msgpack.unpackb(raw_body, raw=False)
                env_key = body_data["env_key"]
                env_id = body_data["env_id"]
                render_mode = body_data.get("render_mode", None)
            except Exception as e:
                raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=f"Invalid msgpack data: {str(e)}")

            response_dict = self.make(env_key=env_key, env_id=env_id, render_mode=render_mode)
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

        @self.app.post("/make_vec")
        async def make_vec_endpoint(request: Request) -> Response:
            """
            Equivalent to /make_vec, but msgpack-based.
            Expects:
              {
                "env_key": str,
                "env_id": str,
                "num_envs": int
              }
            """
            raw_body = await request.body()
            try:
                body_data = msgpack.unpackb(raw_body, raw=False)
                env_key = body_data["env_key"]
                env_id = body_data["env_id"]
                num_envs = int(body_data["num_envs"])
            except Exception as e:
                raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=f"Invalid msgpack data: {str(e)}")

            response_dict = self.make_vec(env_key=env_key, env_id=env_id, num_envs=num_envs)
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

        @self.app.post("/reset")
        async def reset_endpoint(request: Request) -> Response:
            """
            Equivalent to /reset, but msgpack-based.
            Expects:
              {
                "env_key": str,
                "seed": Optional[int],
                "options": Optional[Any]
              }
            """
            raw_body = await request.body()
            try:
                body_data = msgpack.unpackb(raw_body, raw=False)
                env_key = body_data["env_key"]
                seed = body_data.get("seed", None)
                options = body_data.get("options", None)
            except Exception as e:
                raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=f"Invalid msgpack data: {str(e)}")

            response_dict = self.reset(env_key=env_key, seed=seed, options=options)
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

        @self.app.post("/step")
        async def step_endpoint(request: Request) -> Response:
            """
            Equivalent to /step, but msgpack-based.
            Expects:
              {
                "env_key": bytes or str,
                "action": ...
              }
            """
            raw_body = await request.body()
            try:
                body_data = msgpack.unpackb(raw_body, raw=False)
                env_key = body_data["env_key"]
                action_data = body_data["action"]
            except Exception as e:
                raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=f"Invalid msgpack data: {str(e)}")

            # Perform step logic
            response_dict = self.step(env_key=env_key, action_data=action_data)
            
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

        @self.app.get("/action_space")
        async def action_space_endpoint(env_key: str) -> Response:
            """Equivalent to /action_space but returns msgpack."""
            response_dict = self.action_space(env_key)
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

        @self.app.get("/observation_space")
        async def observation_space_endpoint(env_key: str) -> Response:
            """Equivalent to /observation_space but returns msgpack."""
            response_dict = self.observation_space(env_key)
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

        @self.app.post("/close")
        async def close_endpoint(request: Request) -> Response:
            """
            POST /close, receives env_key in the body as msgpack.
            Expects:
            {
                "env_key": str
            }
            """
            raw_body = await request.body()
            try:
                body_data = msgpack.unpackb(raw_body, raw=False)
                env_key = body_data["env_key"]
            except Exception as e:
                raise HTTPException(status_code=HTTP_BAD_REQUEST, detail=f"Invalid msgpack data: {str(e)}")

            response_dict = self.close(env_key)
            packed = msgpack.packb(response_dict, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")

    # ------------------------------------------------
    # The methods each endpoint calls (same logic, but returning Python dicts)
    # ------------------------------------------------
    def make(self, env_key: str, env_id: str, render_mode: Optional[str] = None):
        if not self.env_simulator or not hasattr(self.env_simulator, "make"):
            raise HTTPException(status_code=HTTP_BAD_REQUEST,
                                detail="Backend not properly registered.")
        env_instance = self.env_simulator.make(env_id, render_mode=render_mode)
        self.environments[env_key] = env_instance
        logging.info(f"Environment {env_id} created with key {env_key}.")
        return {
            "message": f"Environment {env_id} created.",    
            "env_key": env_key
        }

    def make_vec(self, env_key: str, env_id: str, num_envs: int):
        if not self.env_simulator or not hasattr(self.env_simulator, "make_vec"):
            raise HTTPException(status_code=HTTP_BAD_REQUEST,
                                detail="Backend not properly registered.")
        env_instance = self.env_simulator.make_vec(env_id, num_envs=num_envs)
        self.environments[env_key] = env_instance
        logging.info(f"Vectorized env {env_id} with {num_envs} instance(s), key {env_key}.")
        return {
            "message": f"Environment {env_id} created with {num_envs} instance(s).",
            "env_key": env_key
        }

    def reset(self, env_key: str, seed: Optional[int], options: Optional[Any]):
        if env_key not in self.environments:
            raise HTTPException(status_code=HTTP_BAD_REQUEST,
                                detail="Environment not initialized. Please call /make first.")
        env = self.environments[env_key]
        observation, info = env.reset(seed=seed, options=options)
        # Convert to nested lists for serialization
        observation, info = (
            convert_ndarrays_to_nested_lists(x) for x in (observation, info)
        )
        return {"observation": observation, "info": info}

    def step(self, env_key: str, action_data):
        # Check environment existence
        if env_key not in self.environments:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        env = self.environments[env_key]
        
        action = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        
        try:
            observation, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            logging.exception("Error in env.step()")
            raise HTTPException(status_code=HTTP_INTERNAL_SERVER_ERROR, detail=str(e))

        # Convert all to nested lists for messagepack
        observation, reward, terminated, truncated, info = (
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
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        action_space = self.environments[env_key].action_space
        action_space = space_to_dict(action_space)
        return replace_nans_infs(action_space)

    def observation_space(self, env_key: str):
        if env_key not in self.environments:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        observation_space = self.environments[env_key].observation_space
        observation_space = space_to_dict(observation_space)
        return replace_nans_infs(observation_space)

    def close(self, env_key: str):
        if env_key not in self.environments:
            # Close all environments if the key is not found
            for key in list(self.environments.keys()):
                self.environments[key].close()
                logging.info(f"Environment with key {key} closed.")
                del self.environments[key]
            return {"message": "All environments closed successfully."}
        
        # Otherwise, close only the specified environment.
        self.environments[env_key].close()
        del self.environments[env_key]
        logging.info(f"Environment with key {env_key} closed.")
        return {"message": f"Environment {env_key} closed successfully."}