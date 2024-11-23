import gymnasium as gym

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
