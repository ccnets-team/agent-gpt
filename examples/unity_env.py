print('Example ML-Agents environment wrapper code here')


from environments.factory import EnvironmentFactory
from unity3d.unity_backend import UnityBackend

# Register UnityBackend
EnvironmentFactory.register("unity", UnityBackend)

# Example usage
if __name__ == "__main__":
    env = EnvironmentFactory.make("unity", file_name="path/to/UnityEnv")
    obs = env.reset()
    print("Initial Observation:", obs)
    env.step([0.1, -0.1])
    env.close()