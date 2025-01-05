# env_hosting/env_api.py
import logging
import numpy as np
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.data_converters import replace_nans_infs

# ------------------------------------------------
# Utility imports
# ------------------------------------------------
from utils.data_converters import (
    convert_ndarrays_to_nested_lists,
    convert_nested_lists_to_ndarrays,
)
from utils.gym_space import space_to_dict

HTTP_BAD_REQUEST = 400
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500

# ------------------------------------------------
# EnvAPI class with FastAPI integration
# ------------------------------------------------
class EnvAPI:
    def __init__(self, env_simulator):
        """
        env_simulator: an object that must have .make(...) and .make_vec(...)
        """
        self.env_simulator = env_simulator
        self.environments = {}

        # Create a FastAPI instance
        self.app = FastAPI()

        # Define all routes
        self._define_endpoints()

    def _define_endpoints(self):
        """Attach all routes/endpoints to self.app."""

        @self.app.post("/make")
        def make_endpoint(body: MakeRequest):
            return self.make(env_id=body.env_id, env_key=body.env_key)

        @self.app.post("/make_vec")
        def make_vec_endpoint(body: MakeVecRequest):
            return self.make_vec(
                env_id=body.env_id,
                env_key=body.env_key,
                num_envs=body.num_envs
            )

        @self.app.post("/reset")
        def reset_endpoint(body: ResetRequest):
            return self.reset(env_key=body.env_key, seed=body.seed, options=body.options)

        @self.app.post("/step")
        def step_endpoint(body: StepRequest):
            return self.step(env_key=body.env_key, action_data=body.action)

        @self.app.get("/action_space")
        def action_space_endpoint(env_key: str):
            return self.action_space(env_key)

        @self.app.get("/observation_space")
        def observation_space_endpoint(env_key: str):
            return self.observation_space(env_key)

        @self.app.post("/close")
        def close_endpoint(env_key: str):
            return self.close(env_key)

    # ------------------------------------------------
    # The methods each endpoint calls
    # ------------------------------------------------
    def make(self, env_id: str, env_key: str):
        """Equivalent to your /make endpoint."""

        if not self.env_simulator or not hasattr(self.env_simulator, "make"):
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Backend not properly registered."
            )

        # Create environment
        env_instance = self.env_simulator.make(env_id)
        self.environments[env_key] = {
            "env": env_instance,
            "is_vectorized": False,
        }
        logging.info(f"Environment {env_id} created with key {env_key}.")
        return {
            "message": f"Environment {env_id} created.",
            "env_key": env_key
        }

    def make_vec(self, env_id: str, env_key: str, num_envs: int):
        """Equivalent to your /make_vec endpoint."""

        if not self.env_simulator or not hasattr(self.env_simulator, "make_vec"):
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Backend not properly registered."
            )

        env_instance = self.env_simulator.make_vec(env_id, num_envs=num_envs)
        self.environments[env_key] = {
            "env": env_instance,
            "is_vectorized": True,
        }
        logging.info(f"Vectorized env {env_id} with {num_envs} instances, key {env_key}.")
        return {
            "message": f"Environment {env_id} created with {num_envs} instance(s).",
            "env_key": env_key
        }

    def reset(self, env_key: str, seed: Optional[int], options: Optional[Any]):
        """Equivalent to your /reset endpoint."""
        if env_key not in self.environments:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        env = self.environments[env_key]["env"]
        observation, info = env.reset(seed=seed, options=options)
        # Convert to nested lists
        observation, info = (
            convert_ndarrays_to_nested_lists(x) for x in (observation, info)
        )
        return {"observation": observation, "info": info}

    def step(self, env_key: str, action_data):
        """Equivalent to your /step endpoint."""
        if env_key not in self.environments:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        env = self.environments[env_key]["env"]
        # Convert Python lists to np.ndarray
        action = convert_nested_lists_to_ndarrays(action_data, dtype=np.float32)
        
        observation, reward, terminated, truncated, info = env.step(action)
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
        """Equivalent to your /action_space endpoint."""
        if env_key not in self.environments:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        action_space = self.environments[env_key]["env"].action_space
        action_space = space_to_dict(action_space)
        return replace_nans_infs(action_space)

    def observation_space(self, env_key: str):
        """Equivalent to your /observation_space endpoint."""
        if env_key not in self.environments:
            raise HTTPException(
                status_code=HTTP_BAD_REQUEST,
                detail="Environment not initialized. Please call /make first."
            )
        observation_space = self.environments[env_key]["env"].observation_space
        observation_space = space_to_dict(observation_space)
        return replace_nans_infs(observation_space)

    def close(self, env_key: str):
        """Equivalent to your /close endpoint."""
        if not env_key:
            # close all
            for key in list(self.environments.keys()):
                self.environments[key]["env"].close()
                del self.environments[key]
            return {"message": "All environments closed."}

        if env_key in self.environments:
            self.environments[env_key]["env"].close()
            del self.environments[env_key]
            logging.info(f"Environment with key {env_key} closed.")
            return {"message": f"Environment {env_key} closed successfully."}
        
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="No environment with this key to close."
        )

# ------------------------------------------------
# Pydantic request models
# ------------------------------------------------
class MakeRequest(BaseModel):
    env_id: str
    env_key: str

class MakeVecRequest(BaseModel):
    env_id: str
    env_key: str
    num_envs: int = 1

class ResetRequest(BaseModel):
    env_key: str
    seed: Optional[int] = None
    options: Optional[Any] = None

class StepRequest(BaseModel):
    env_key: str
    action: Any  # We'll convert it to np.ndarray