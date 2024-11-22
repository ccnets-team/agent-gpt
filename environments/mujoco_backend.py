import gymnasium as gym

class MujocoBackend:
    def __init__(self, env):
        """Initialize the backend."""
        self.env = env

    @staticmethod
    def make(env_id, **kwargs):
        """Create a single environment."""
        return gym.make(env_id, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        """Create a vectorized environment."""
        return gym.make_vec(env_id, num_envs = num_envs, **kwargs)

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)

    def close(self):
        """Close the environment."""
        self.env.close()
    