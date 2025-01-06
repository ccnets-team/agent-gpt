# env_wrapper/gym_env.py
import gymnasium as gym
from gymnasium.envs import registration

class GymEnv:
    def __init__(self, env, **kwargs):
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
        
    @classmethod
    def register(cls, id, entry_point):
        
        # Dynamically retrieve module and class name
        module_path = cls.__module__    
        class_name = cls.__name__
        
        # Handle `None` for env_id naturally
        if id is None:
            id = f"{class_name}"  # Use a default name if env_id is None

        # Additional logic to register the environment
        print(f"Registering environment: {id} with API URL: {entry_point}")

        # Register environment with dynamic module path and class name, and pass env_endpoint via kwargs
        registration.register(
            id=id,
            entry_point=f"{module_path}:{class_name}",
            kwargs={"entry_point": entry_point, "id": id}  # Add env_id to kwargs
        )