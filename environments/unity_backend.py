import numpy as np
import logging
from gymnasium import Env, spaces
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class UnityBackend(Env):
    def __init__(self, env_id, num_envs=1, use_graphics=False, is_vectorized=False, time_scale=20, **kwargs):
        """
        Initialize a Unity environment.

        :param env_id: Path to the Unity environment binary.
        :param num_envs: Number of environments (for vectorized environments).
        :param worker_id: Unique identifier for multiple Unity instances.
        :param use_graphics: Whether to run with graphics.
        :param is_vectorized: Whether the environment is vectorized.
        :param seed: Random seed for the environment.
        :param time_scale: Time scale for the Unity environment.
        """
        super().__init__()
        
        self.seed = kwargs.get("seed", 0)
        self.worker_id = self.seed
        self.env_id = env_id
        self.num_envs = num_envs
        self.is_vectorized = is_vectorized 
        self.no_graphics = not use_graphics
        self.file_name = self.file_name = "../unity_environments/" + "3DBallHard" +"/"
        self.time_scale = time_scale
        self.channel = EngineConfigurationChannel()
        self.channel.set_configuration_parameters(width=1280, height=720, time_scale=self.time_scale)

        if is_vectorized:
            # Create multiple environments without graphics for performance
            self.envs = [
                self.create_unity_env(
                    self.file_name,
                    self.channel,
                    no_graphics=True,
                    seed=self.seed + i,
                    worker_id=self.worker_id + i
                ) for i in range(num_envs)
            ]
        else:
            self.env = self.create_unity_env(
                self.file_name,
                self.channel,
                no_graphics=self.no_graphics,
                seed=self.seed,
                worker_id=self.worker_id
            )
            self.envs = [self.env]  # For consistency, make self.envs a list

        self.behavior_names = []
        self.specs = []
        self.agent_per_envs = [] 
        self.num_agents = 0
        self.from_local_to_global = []
        self.env_agent_offsets = []  # Offsets for agent indices in each env
        self.decision_agents = []
        self._setup_spaces()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.envs:
            self.close()
        
    @staticmethod
    def create_unity_env(file_name, channel, no_graphics, seed, worker_id):
        base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT + worker_id
        env = UnityEnvironment(
            file_name=file_name,
            base_port=base_port,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[channel],
            worker_id=worker_id,
        )        
        return env

    def _setup_spaces(self):
        total_agents = 0
        # Collect behavior names, specs, and agent counts for each environment
        for env in self.envs:
            env.reset()
            behavior_name = list(env.behavior_specs.keys())[0]
            self.behavior_names.append(behavior_name)
            self.specs.append(env.behavior_specs[behavior_name])

            # Get initial agent IDs and count
            decision_steps, _ = env.get_steps(behavior_name)
            num_agents = len(decision_steps)
            self.agent_per_envs.append(num_agents)
            self.env_agent_offsets.append(total_agents)
            env.reset() # Reset the environment again before starting the episode

            self.decision_agents.append(np.zeros(num_agents, dtype=np.bool_))  
            # Create mapping from local to global indices
            local_to_global = []
            for local_idx in range(num_agents):
                global_idx = total_agents + local_idx
                local_to_global.append(global_idx)
            self.from_local_to_global.append(local_to_global)

            total_agents += num_agents

        self.num_agents = total_agents

        # Assume all environments have the same action and observation spaces
        self.spec = self.specs[0]

        # Define the action space
        action_spec = self.spec.action_spec
        if action_spec.is_continuous():
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_agents, action_spec.continuous_size),
                dtype=np.float32,
            )
        elif action_spec.is_discrete():
            # Calculate the total number of possible actions for all branches
            total_discrete_actions = np.prod(action_spec.discrete_branches)
            self.action_space = spaces.Discrete(total_discrete_actions)
        else:
            # Mixed action space (not fully supported in this example)
            raise NotImplementedError("Mixed action spaces are not supported in this implementation.")

        # Define the observation space
        observation_shapes = [obs_spec.shape for obs_spec in self.spec.observation_specs]

        # Helper function to determine if a shape is an image
        def is_image(shape):
            """An image is assumed to have at least 3 dimensions, e.g., (H, W, C)."""
            return len(shape) == 3

        # Check if any observation shape is an image
        if any(is_image(shape) for shape in observation_shapes):
            raise ValueError("Image observations are not supported.")

        # Combine shapes by summing up their first dimensions
        self.observation_shapes =  observation_shapes
        self.combined_dim = sum(np.prod(shape) for shape in observation_shapes)

        # Assuming `self.num_agents` and `self.combined_dim` are already defined
        low = np.full((self.num_agents, self.combined_dim), -np.inf, dtype=np.float32)
        high = np.full((self.num_agents, self.combined_dim), np.inf, dtype=np.float32)

        # Define the observation space
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self.num_agents, self.combined_dim),
            dtype=np.float32,
        )        

    @staticmethod
    def make(env_id, **kwargs):
        return UnityBackend(env_id=env_id, is_vectorized=False, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        return UnityBackend(env_id=env_id, num_envs=num_envs, is_vectorized=True, **kwargs)

    def reset(self, **kwargs):
        """
        Reset the Unity environment(s) and retrieve initial observations.
        :return: Initial aggregated observations and info dictionary.
        """
        try:
            observations = [None] * self.num_agents
            for env_idx, env in enumerate(self.envs):
                env.reset()
                behavior_name = self.behavior_names[env_idx]
                decision_steps, _ = env.get_steps(behavior_name)

                self.decision_agents[env_idx] = np.zeros_like(self.decision_agents[env_idx])
                self.decision_agents[env_idx][decision_steps.agent_id] = True

                obs = self.aggregate_observations(decision_steps.obs)
                for idx, agent_id in enumerate(decision_steps.agent_id):
                    global_idx = self.from_local_to_global[env_idx][agent_id]
                    # Aggregate all observation components
                    observations[global_idx] = obs[idx]

            return observations, {}
        except Exception as e:
            logging.error(f"An error occurred during reset: {e}")
            raise e
    
    def init_transitions(self):
        return  [None] * self.num_agents, [None] * self.num_agents, [None] * self.num_agents, [None] * self.num_agents, [None] * self.num_agents
    
    def step(self, actions):
        """
        Perform a step in the Unity environment(s).
        :param actions: Actions to take for all agents.
        :return: Tuple containing observations, rewards, terminated flags, truncated flags, and info.
        """
        try:
            action_offset = 0
            
            # Set actions for all environments
            for env_idx, env in enumerate(self.envs):
                num_agents_in_env = self.agent_per_envs[env_idx]
                env_actions = actions[action_offset:action_offset + num_agents_in_env]
                action_offset += num_agents_in_env
                
                decision_check = self.decision_agents[env_idx]
                dec_actions = env_actions[decision_check]
                
                action_tuple = self._create_action_tuple(dec_actions, env_idx)
                env.set_actions(self.behavior_names[env_idx], action_tuple)
                env.step()

        except Exception as e:
            logging.error(f"An error occurred during the step: {e}")
            raise e
        try:
            observations, rewards, terminated, truncated, final_observations = self.init_transitions()
            # Collect results from all environments
            for env_idx, env in enumerate(self.envs):
                decision_steps, terminal_steps = env.get_steps(self.behavior_names[env_idx])
                self.decision_agents[env_idx] = np.zeros_like(self.decision_agents[env_idx])
                self.decision_agents[env_idx][decision_steps.agent_id] = True

                # Get agent IDs and mapping from agent_id to local index
                decision_agent_id_to_local = {agent_id: idx for idx, agent_id in enumerate(decision_steps.agent_id)}
                terminal_agent_id_to_local = {agent_id: idx for idx, agent_id in enumerate(terminal_steps.agent_id)}

                # Agents present in both decision and terminal steps
                common_agent_ids = set(decision_steps.agent_id).intersection(terminal_steps.agent_id)

                # Agents only in decision steps
                decision_only_agent_ids = set(decision_steps.agent_id) - common_agent_ids

                # Agents only in terminal steps
                terminal_only_agent_ids = set(terminal_steps.agent_id) - common_agent_ids

                # Handle agents present in both decision and terminal steps
                dec_obs = self.aggregate_observations(decision_steps.obs)
                term_obs = self.aggregate_observations(terminal_steps.obs)
                for agent_id in common_agent_ids:
                    local_idx = decision_agent_id_to_local[agent_id]
                    global_idx = self.from_local_to_global[env_idx][agent_id]
                    # Aggregate observations
                    term_local_idx = terminal_agent_id_to_local[agent_id]
                    final_observations[global_idx] = term_obs[term_local_idx]
                    observations[global_idx] = dec_obs[local_idx]
                    rewards[global_idx] = float(decision_steps.reward[local_idx])
                    terminated[global_idx] = True
                    truncated[global_idx] = False

                # Handle agents only in decision steps
                for agent_id in decision_only_agent_ids:
                    local_idx = decision_agent_id_to_local[agent_id]
                    global_idx = self.from_local_to_global[env_idx][local_idx]
                    observations[global_idx] = dec_obs[local_idx]
                    rewards[global_idx] = float(decision_steps.reward[local_idx])
                    terminated[global_idx] = False
                    truncated[global_idx] = False

                # Handle agents only in terminal steps
                for agent_id in terminal_only_agent_ids:
                    local_idx = terminal_agent_id_to_local[agent_id]
                    global_idx = self.from_local_to_global[env_idx][local_idx]
                    observations[global_idx] = term_obs[local_idx]
                    rewards[global_idx] = float(terminal_steps.reward[local_idx])
                    terminated[global_idx] = True
                    truncated[global_idx] = False  # Adjust if necessary

        except Exception as e:
            logging.error(f"An error occurred during the step: {e}")
            raise e
    
        info = {}
        info['final_observation'] = final_observations
            
        return observations, rewards, terminated, truncated, info

    def aggregate_observations(self, observations):
        """
        Combine observations from multiple shapes into a single aggregated vector.
        """
        num_agents = len(observations[0])
        combined_dim = sum(np.prod(shape) for shape in self.observation_shapes)
        aggregated = np.zeros((num_agents, combined_dim), dtype=np.float32)
        if num_agents < 1:
            return aggregated
        offset = 0

        for obs, shape in zip(observations, self.observation_shapes):
            size = np.prod(shape)
            obs = obs.reshape(num_agents, -1)
            aggregated[:, offset:offset + size] = obs  # Flatten the observation and insert
            offset += size

        return aggregated
    
    def _create_action_tuple(self, actions, env_idx):
        action_spec = self.specs[env_idx].action_spec
        action_tuple = ActionTuple()
        num_agents = len(actions)

        if action_spec.is_continuous():
            # Ensure actions are in the correct shape
            if isinstance(actions, list):
                actions = np.array(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape((num_agents, -1))
            action_tuple.add_continuous(actions)
        elif action_spec.is_discrete():
            # Ensure actions are in the correct shape
            if isinstance(actions, list):
                actions = np.array(actions, dtype=np.int32)
            if actions.ndim == 1:
                actions = actions.reshape((num_agents, -1))
            action_tuple.add_discrete(actions)
        else:
            raise NotImplementedError("Mixed action spaces are not supported in this implementation.")

        return action_tuple

    def close(self):
        """
        Close the Unity environment(s).
        """
        for env in self.envs:
            env.close()
            
        self.envs = []
