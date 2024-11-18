from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np

class UnityBackend:
    def __init__(self, file_name, base_port=5005, no_graphics=True, seed=0, time_scale=20):
        """
        Initialize a Unity environment.
        :param file_name: Path to the Unity environment binary.
        :param base_port: Base port for the environment connection.
        :param no_graphics: Whether to run without graphics.
        :param seed: Random seed for the environment.
        :param time_scale: Time scale for the Unity environment.
        """
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(time_scale=time_scale)
        self.env = UnityEnvironment(
            file_name=file_name,
            base_port=base_port,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[self.channel],
        )
        self.behavior_name = None

    @staticmethod
    def make(env_id, **kwargs):
        """
        Static method to create a Unity environment.
        :param env_id: Path to the Unity environment binary.
        :param kwargs: Additional configuration options (e.g., base_port, no_graphics, time_scale).
        """
        return UnityBackend(file_name=env_id, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        """
        Static method to create a vectorized Unity environment.
        Note: Unity does not natively support vectorized environments. This is a placeholder.
        :param env_id: Path to the Unity environment binary.
        :param num_envs: Number of environments to simulate (not supported).
        """
        raise NotImplementedError("Vectorized environments are not supported for Unity.")

    def reset(self):
        """
        Reset the Unity environment and retrieve initial observations.
        :return: Initial observations for all agents.
        """
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        return decision_steps.obs

    def step(self, actions):
        """
        Perform a step in the Unity environment.
        :param actions: Actions to take for all agents.
        :return: Tuple containing observations, rewards, done flags, and info.
        """
        action_tuple = ActionTuple()
        if isinstance(actions, np.ndarray) and actions.ndim == 2:
            action_tuple.add_continuous(actions)
        else:
            raise ValueError("Actions must be a 2D NumPy array for Unity environments.")

        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        obs = decision_steps.obs
        rewards = decision_steps.reward
        done = len(terminal_steps) > 0
        info = {"terminal_rewards": terminal_steps.reward} if done else {}

        return obs, rewards, done, info

    def close(self):
        """
        Close the Unity environment.
        """
        self.env.close()
