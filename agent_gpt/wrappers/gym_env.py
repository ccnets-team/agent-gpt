# env_wrapper/gym_env.py
import gymnasium as gym

class GymEnv:
    def __init__(self, env, **kwargs):
        """Initialize the backend."""
        self.env = env
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
            
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
        if self.env:
            self.env.close()
            self.env = None
        
    @classmethod
    def register(cls, env_id, entry_point):
        from gymnasium.envs import registration
        from gymnasium.error import UnregisteredEnv  # For older versions, it might be gym.error.Error
        import gymnasium
        try:
            # If the spec is found, the environment is already registered.
            gymnasium.spec(env_id)
            print(f"Environment {env_id} is already registered; skipping registration.")
        except UnregisteredEnv:
            print(f"Registering Gym environment: {env_id} with entry_point: {entry_point}")
            try:
                registration.register(
                    id=env_id,
                    entry_point=entry_point,
                    kwargs={"entry_point": entry_point, "id": env_id}
                )        
            except Exception as e:
                print(f"Error registering environment {env_id}: {e}")
                raise e