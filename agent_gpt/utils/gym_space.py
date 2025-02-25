# utils/gym_space.py
import numpy as np
import gymnasium as gym

"""
This module defines two reversible functions for converting Gymnasium spaces to
and from Python dictionaries. Gym spaces often contain NumPy arrays (e.g., Box
bounds), which aren’t directly JSON-friendly for HTTP communication with remote
trainers. By serializing these arrays into lists, we can send them over the wire
and then reconstruct the exact space object at the destination.

Functions:
  - space_to_dict(space): Recursively serialize a Gym space into a Python dict,
                          converting NumPy arrays into lists.
  - space_from_dict(data): Recursively deserialize that dict back into the
                          original Gym space, restoring the NumPy arrays.
"""

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
