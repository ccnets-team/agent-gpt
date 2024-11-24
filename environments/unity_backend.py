import numpy as np
import logging
from gymnasium import Env, spaces
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class UnityBackend(Env):
    def __init__(self, env_id, num_envs=1, use_graphics=False, is_vectorized=False, time_scale=128, **kwargs):
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
        self.file_name = self.file_name = "../unity_environments/" + "PushBlock" +"/"
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
        self.decision_agents = []
        self._initialize_env_info()
        self._define_observation_space()
        self._define_action_space()
    
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

    @staticmethod
    def make(env_id, **kwargs):
        return UnityBackend(env_id=env_id, is_vectorized=False, **kwargs)

    @staticmethod
    def make_vec(env_id, num_envs, **kwargs):
        return UnityBackend(env_id=env_id, num_envs=num_envs, is_vectorized=True, **kwargs)
    
    def _initialize_env_info(self):
        total_agents = 0
        # Collect behavior names, specs, and agent counts for each environment
        for env in self.envs:
            env.reset()
            behavior_name = list(env.behavior_specs.keys())[0]
            self.behavior_names.append(behavior_name)
            self.specs.append(env.behavior_specs[behavior_name])

            # Get initial agent IDs and count
            decision_steps, _ = env.get_steps(behavior_name)
            n_agents = len(decision_steps)
            self.agent_per_envs.append(n_agents)
            env.reset()  # Reset the environment again before starting the episode

            self.decision_agents.append(np.zeros(n_agents, dtype=np.bool_))

            # Create mapping from local to global indices
            local_to_global = []
            for local_idx in range(n_agents):
                global_idx = total_agents + local_idx
                local_to_global.append(global_idx)
            self.from_local_to_global.append(local_to_global)

            total_agents += n_agents

        self.num_agents = total_agents

    def _define_observation_space(self):
        # Define the observation space per agent
        observation_shapes = [obs_spec.shape for obs_spec in self.specs[0].observation_specs]

        # Check if any observation shape is an image
        if any(len(shape) == 3 for shape in observation_shapes):
            raise ValueError("Image observations are not supported.")

        self.observation_shapes = observation_shapes
        self.combined_dim = sum(np.prod(shape) for shape in observation_shapes)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.combined_dim),
            dtype=np.float32,
        )

    def _define_action_space(self):
        # Assume all environments have the same action and observation spaces
        self.spec = self.specs[0]
        action_spec = self.spec.action_spec

        # Define the action space per agent
        if action_spec.continuous_size > 0 and action_spec.discrete_size == 0:
            # Continuous actions only
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
        elif action_spec.discrete_size > 0 and action_spec.continuous_size == 0:
            if action_spec.discrete_size == 1:
                # Single discrete action branch
                self.action_space = spaces.Discrete(action_spec.discrete_branches[0]-1, start=1)
            else:
                # Multiple discrete action branches
                self.action_space = spaces.MultiDiscrete(action_spec.discrete_branches)
        elif action_spec.continuous_size > 0 and action_spec.discrete_size > 0:
            # Mixed actions
            # Use spaces.Tuple to combine continuous and discrete action spaces
            if action_spec.discrete_size == 1:
                discrete_space = spaces.Discrete(action_spec.discrete_branches[0])
            else:
                discrete_space = spaces.MultiDiscrete(action_spec.discrete_branches)
            self.action_space = spaces.Tuple((
                spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(action_spec.continuous_size,),
                    dtype=np.float32,
                ),
                discrete_space,
            ))
        else:
            raise NotImplementedError("Action space not supported.")

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
        num_agents = self.num_agents
        observations = [None] * num_agents
        rewards = [None] * num_agents
        terminated = [None] * num_agents
        truncated = [None] * num_agents
        final_observations = [None] * num_agents
        return observations, rewards, terminated, truncated, final_observations
    
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
                if len(dec_actions) > 0:
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
                    dec_local_idx = decision_agent_id_to_local[agent_id]
                    term_local_idx = terminal_agent_id_to_local[agent_id]
                    global_idx = self.from_local_to_global[env_idx][agent_id]
                    # Aggregate observations
                    final_observations[global_idx] = term_obs[term_local_idx]
                    observations[global_idx] = dec_obs[dec_local_idx]
                    rewards[global_idx] = float(terminal_steps.reward[term_local_idx])
                    terminated[global_idx] = True
                    truncated[global_idx] = False

                # Handle agents only in decision steps
                for agent_id in decision_only_agent_ids:
                    dec_local_idx = decision_agent_id_to_local[agent_id]
                    global_idx = self.from_local_to_global[env_idx][agent_id]
                    observations[global_idx] = dec_obs[dec_local_idx]
                    rewards[global_idx] = float(decision_steps.reward[dec_local_idx])
                    terminated[global_idx] = False
                    truncated[global_idx] = False

                # Handle agents only in terminal steps
                for agent_id in terminal_only_agent_ids:
                    dec_local_idx = terminal_agent_id_to_local[agent_id]
                    global_idx = self.from_local_to_global[env_idx][agent_id]
                    observations[global_idx] = term_obs[dec_local_idx]
                    rewards[global_idx] = float(terminal_steps.reward[dec_local_idx])
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

        if num_agents == 0:
            # No agents to act upon
            return action_tuple

        if isinstance(self.action_space, spaces.Box):
            # Continuous actions only
            actions = np.asarray(actions, dtype=np.float32).reshape(num_agents, -1)
            action_tuple.add_continuous(actions)
        elif isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete):
            # Discrete actions only
            actions = np.asarray(actions, dtype=np.int32).reshape(num_agents, -1)
            action_tuple.add_discrete(actions)
        elif isinstance(self.action_space, spaces.Tuple):
            # Mixed actions: actions are tuples (continuous_action, discrete_action)
            continuous_actions = []
            discrete_actions = []

            for action in actions:
                continuous_action, discrete_action = action  # Unpack the tuple
                continuous_actions.append(continuous_action)
                discrete_actions.append(discrete_action)

            continuous_actions = np.asarray(continuous_actions, dtype=np.float32).reshape(num_agents, -1)
            discrete_actions = np.asarray(discrete_actions, dtype=np.int32).reshape(num_agents, -1)

            action_tuple.add_continuous(continuous_actions)
            action_tuple.add_discrete(discrete_actions)
        else:
            raise NotImplementedError("Action type not supported.")

        return action_tuple

    def close(self):
        """
        Close the Unity environment(s).
        """
        for env in self.envs:
            env.close()
            
        self.envs = []
