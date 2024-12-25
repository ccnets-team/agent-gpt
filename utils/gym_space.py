# gym_space.py

import numpy as np
import gymnasium as gym

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
    
def compute_space_dimension(observation_space, is_vectorized):
    if isinstance(observation_space, gym.spaces.Box):
        # Flatten all dimensions
        if is_vectorized:
            return int(np.prod(observation_space.shape[1:]))
        else:
            return int(np.prod(observation_space.shape))

    elif isinstance(observation_space, gym.spaces.Discrete):
        return observation_space.n
    
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return sum(observation_space.nvec)

    elif isinstance(observation_space, gym.spaces.Tuple):
        return sum(compute_space_dimension(subspace, is_vectorized) for subspace in observation_space.spaces)

    elif isinstance(observation_space, gym.spaces.Dict):
        # Use .spaces attribute to get the underlying dictionary of subspaces
        return sum(compute_space_dimension(subspace, is_vectorized) for subspace in observation_space.spaces.values())
    else:
        raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

def serialize_space(space):
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": space.low.tolist(),
            "high": space.high.tolist(),
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
            "start": int(space.start),
        }
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": space.nvec.tolist(),
            "start": space.start.tolist(),
        }
    elif isinstance(space, gym.spaces.Tuple):
        return {
            "type": "Tuple",
            "spaces": [serialize_space(s) for s in space.spaces],
        }
    elif isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {key: serialize_space(s) for key, s in space.spaces.items()},
        }
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")
