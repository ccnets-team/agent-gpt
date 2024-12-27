# gym_space.py

import numpy as np
import gymnasium as gym
from typing import Any, Dict

def flatten_observations(data, num_agents):
    """
    Flatten the observation data structure into (num_agents, feature_dim).
    
    This function handles nested (dict, tuple, list) structures by concatenating their parts.
    For single arrays or scalars, it ensures the shape is (num_agents, -1).
    """
    if isinstance(data, (dict, tuple, list)):
        arrays = []
        for obs in data:
            if isinstance(obs, (dict, tuple, list)):
                arr = flatten_observations(obs, num_agents)
            else:
                arr = np.array(obs, dtype=np.float32).reshape(num_agents, -1)
            arrays.append(arr)
        return np.concatenate(arrays, axis=-1)
    else:
        # Handle scalar or simple array directly
        arr = np.array(data, dtype=np.float32)
        return arr.reshape(num_agents, -1)    

def compute_space_dimension(observation_space):
    if isinstance(observation_space, gym.spaces.Box):
        return int(observation_space.shape[-1])

    elif isinstance(observation_space, gym.spaces.Discrete):
        return observation_space.n
    
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return sum(observation_space.nvec)

    elif isinstance(observation_space, gym.spaces.Tuple):
        return sum(compute_space_dimension(subspace) for subspace in observation_space.spaces)

    elif isinstance(observation_space, gym.spaces.Dict):
        # Use .spaces attribute to get the underlying dictionary of subspaces
        return sum(compute_space_dimension(subspace) for subspace in observation_space.spaces.values())
    else:
        raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

def space_to_dict(space: gym.spaces.Space):
    """Recursively serialize a Gym space into a Python dict."""
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "low": space.low.tolist(),   # convert np.ndarray -> list
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype)
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete",
            "n": space.n
        }
    elif isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                k: space_to_dict(v) for k, v in space.spaces.items()
            }
        }
    elif isinstance(space, gym.spaces.Tuple):
        return {
            "type": "Tuple",
            "spaces": [space_to_dict(s) for s in space.spaces]
        }
    else:
        raise NotImplementedError(f"Cannot serialize space type: {type(space)}")

def space_from_dict(data: dict) -> gym.spaces.Space:
    """Recursively deserialize a Python dict to a Gym space."""
    space_type = data["type"]
    if space_type == "Box":
        low = np.array(data["low"], dtype=float)
        high = np.array(data["high"], dtype=float)
        shape = tuple(data["shape"])
        dtype = data.get("dtype", "float32")
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    elif space_type == "Discrete":
        return gym.spaces.Discrete(data["n"])
    elif space_type == "Dict":
        sub_dict = {
            k: space_from_dict(v) for k, v in data["spaces"].items()
        }
        return gym.spaces.Dict(sub_dict)
    elif space_type == "Tuple":
        sub_spaces = [space_from_dict(s) for s in data["spaces"]]
        return gym.spaces.Tuple(tuple(sub_spaces))
    else:
        raise NotImplementedError(f"Cannot deserialize space type: {space_type}")
