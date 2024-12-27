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

def serialize_space(
    space: gym.spaces.Space,
    type_key: str = "space_type",  # e.g. observation_space_type or action_space_type
    value_key: str = "space_value" # e.g. observation_space_value or action_space_value
) -> Dict[str, Any]:
    """
    Serialize a gym space's sample data into a dictionary suitable for JSON storage.
    
    - type_key: Key under which the space's class name will be stored.
    - value_key: Key under which the space's JSON-serialized data will be stored.
    """
    # 1) Extract the space class name, e.g. "Box", "Discrete", etc.
    space_type_str = type(space).__name__

    # 2) Get the JSON-friendly data from to_jsonable.
    #    Note: 'space.to_jsonable' typically expects a *list of samples*,
    #    so if you only have a single sample, wrap it in a list before calling this
    #    or ensure you've already passed a list of samples to 'space'.
    space_value = space.to_jsonable()

    # 3) Return a dict containing both the type and the value
    return {
        type_key: space_type_str,
        value_key: space_value
    }
    
def deserialize_space(
    data: dict, 
    type_key: str = "space_type", # observation_space_type or action_space_type
    value_key: str = "space_value" # observation_space_value or action_space_value
):
    # 1) Extract the space class name
    space_type_str = data.get(type_key)
    if space_type_str is None:
        raise ValueError(f"No '{type_key}' found in data.")

    # 2) Dynamically get the gym.spaces class
    space_class: gym.spaces.OneOf = getattr(gym.spaces, space_type_str, None)
    if space_class is None:
        raise ValueError(f"Unsupported space class: {space_type_str}")

    # 3) Extract the JSON-serialized samples
    space_value = data.get(value_key)
    if space_value is None:
        raise ValueError(f"No '{value_key}' found in data.")

    # 4) Convert from JSONable -> original sample(s)
    deserialized_space = space_class.from_jsonable(space_value)

    return deserialized_space