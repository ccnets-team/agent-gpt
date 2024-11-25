try:
    import gymnasium as gym
except ImportError:
    gym = None
    print("Gymnasium not installed. Please install it to use the Gym backend.")

from environments.environment_factory import EnvironmentFactory

class GymBackend(gym):
    pass

# Register GymBackend
EnvironmentFactory.register("gym", GymBackend)

# Example usage
if __name__ == "__main__":
    env = EnvironmentFactory.make("gym", env_id="CartPole-v1")
    obs = env.reset()
    print("Initial Observation:", obs)
    obs, reward, done, info = env.step(env.action_space.sample())
    print("Step Result:", obs, reward, done, info)
    env.close()