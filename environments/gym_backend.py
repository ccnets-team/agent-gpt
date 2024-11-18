class EnvironmentBackend:
    def __init__(self, **kwargs):
        """Initialize the backend."""
        raise NotImplementedError

    @staticmethod
    def make(env_id, **kwargs):
        """Create an environment."""
        raise NotImplementedError

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        """Create a vectorized environment."""
        raise NotImplementedError

    def reset(self, **kwargs):
        """Reset the environment."""
        raise NotImplementedError

    def step(self, action):
        """Take a step in the environment."""
        raise NotImplementedError

    def close(self):
        """Close the environment."""
        raise NotImplementedError
