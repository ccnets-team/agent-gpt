# custom_env.py
from gymnasium import Env, spaces
import numpy as np

class CustomEnv(Env):
    """
    A minimal custom environment that can be used as an env_type 
    in the EnvGateway, mirroring the structure of UnityEnv and GymEnv.
    """

    def __init__(self, env_id="CustomEnv-v0", **kwargs):
        """
        Initialize your custom environment here.
        :param env_id: A string identifier for the environment.
        :param kwargs: Any extra configuration parameters for the environment.
        """
        super().__init__()

        # Example: define a simple observation & action space
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # e.g., 5 possible discrete actions

        # Store environment ID and kwargs if needed
        self.env_id = env_id
        self.config = kwargs

        # Internal state
        self.state = None
        self.episode_length = 0
        self.max_episode_length = self.config.get("max_episode_length", 50)

    @staticmethod
    def make(env_id, **kwargs):
        """
        Create a single instance of the custom environment.
        :param env_id: A string identifier for the environment.
        :param kwargs: Extra configuration parameters to pass to the constructor.
        :return: An instance of CustomEnv.
        """
        return CustomEnv(env_id=env_id, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        """
        Create a vectorized environment. 
        For a custom solution, you could implement or wrap 
        multiple CustomEnv instances in parallel. 
        Below is a simplistic placeholder example.

        :param env_id: String identifier.
        :param num_envs: Number of parallel environments.
        :param kwargs: Extra configuration parameters.
        :return: A list (or a wrapper) of multiple CustomEnv instances.
        """
        envs = [CustomEnv(env_id=env_id, **kwargs) for _ in range(num_envs)]
        return envs

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation.
        :param seed: An optional random seed.
        :param options: Additional reset options.
        :return: observation, info
        """
        super().reset(seed=seed)

        # Example state initialization
        self.state = np.zeros(shape=(3,), dtype=np.float32)  # could be random or fixed
        self.episode_length = 0

        # `info` dictionary can hold debugging or other info
        info = {}
        return self.state, info

    def step(self, action):
        """
        Take a step in the environment using the given action.
        :param action: The chosen action.
        :return: observation, reward, terminated, truncated, info
        """
        self.episode_length += 1

        # Example transition logic:
        #   - Update state
        #   - Calculate reward
        #   - Check if terminated or truncated
        self.state += np.random.normal(0, 1.0, size=(3,))  # random walk
        reward = float(np.random.rand())  # random reward for demonstration

        # A simple terminal condition
        terminated = bool(self.episode_length >= self.max_episode_length)
        truncated = False  # set True if you have a non-episode-based cutoff

        info = {}
        return self.state, reward, terminated, truncated, info

    def close(self):
        """
        Close the environment.
        """
        # Perform any cleanup here
        pass

    @classmethod
    def register(cls, env_id, entry_point):
        """
        If you’d like to integrate a registry mechanism, 
        similar to `gym.register`, you can implement it here.
        """
        # You can add logic to handle your own registration if needed.
        print(f"Registering custom environment: {env_id} at {entry_point}")
