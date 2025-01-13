# config/hyperparams.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EnvHost:
    """
    Holds the env_endpoint info and agent count for a single hosting environment 
    (whether it's local or remote).
    """
    host_id: Optional[str] = None  # e.g., "local", "remote", "aws", "azure", etc.
    env_endpoint: str                  # e.g., "http://localhost:8000" or "http://ec2-xxx.compute.amazonaws.com"
    num_agents: int = 64

@dataclass
class Exploration:
    """
    Defines exploration parameters for a single action type 
    (continuous or discrete).
    """
    type: str = "gaussian_noise" # "none", "epsilon_greedy", "gaussian_noise", "ornstein_uhlenbeck", "parameter_noise"

    # EpsilonGreedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01

    # GaussianNoise
    initial_sigma: float = 0.1
    final_sigma: float = 0.001

    # OrnsteinUhlenbeckNoise
    mu: float = 0.0
    theta: float = 0.15
    ou_sigma: float = 0.2
    dt: float = 1e-2

    # ParameterNoise
    initial_stddev: float = 0.05
    final_stddev: float = 0.0005
    
    def _fields_for_type(self) -> list[str]:
        """Returns a list of field names relevant to the specified exploration type."""
        if self.type == "none":
            return ["type"]
        elif self.type == "epsilon_greedy":
            return ["type", "initial_epsilon", "final_epsilon"]
        elif self.type == "gaussian_noise":
            return ["type", "initial_sigma", "final_sigma"]
        elif self.type == "ornstein_uhlenbeck":
            return ["type", "mu", "theta", "ou_sigma", "dt"]
        elif self.type == "parameter_noise":
            return ["type", "initial_stddev", "final_stddev"]
        else:
            raise ValueError(f"Invalid exploration type: '{self.type}'")

    def __post_init__(self):
        """
        After the dataclass is initialized, blank out any fields that are not 
        relevant to the chosen exploration type.
        """
        try:
            fields_for_type = self._fields_for_type()
        except ValueError as e:
            # If user provided an invalid exploration type, we won't prune anything
            print("[WARNING]", e)
            return

        # For each field in this object, set it to None if it's not relevant
        for field_name in vars(self):
            if field_name not in fields_for_type:
                setattr(self, field_name, None)
        
@dataclass
class Hyperparameters:
    """
    A single, consolidated dataclass holding hyperparameters and configurations
    for your RL system (environment, training, optimization, etc.).
    """
    # 1) Client / Env
    env_id: Optional[str] = None
    env_hosts: dict[str, EnvHost] = field(default_factory=dict)
    use_tensorboard: bool = True
    use_cloudwatch: bool = True

    # 2) Session
    use_print: bool = True
    use_graphics: bool = False
    max_test_episodes: int = 100

    # 3) Training
    batch_size: int = 64
    train_interval: int = 1
    max_steps: int = 500_000
    buffer_size: int = 500_000

    # 4) Algorithm
    gamma_init: float = 0.99
    lambda_init: float = 0.95
    max_input_states: int = 16
    exploration: dict[str, Exploration] = field(default_factory=dict)

    # 5) Optimization
    lr_init: float = 1e-4
    lr_end: float = 1e-6
    lr_scheduler: str = "exponential"  # "linear", "exponential",
    lr_cycle_steps: int = 20_000
    tau: float = 0.01
    max_grad_norm: float = 1.0

    # 6) Network
    gpt_type: str = "gpt2"  
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.1
    num_heads: int = 8
    
    # -----------------------
    # Methods
    # -----------------------
    def set_exploration(self, key: str, exploration: Exploration):
        """Sets exploration config under a named key, e.g. 'continuous' or 'discrete'."""
        assert key in ["continuous", "discrete"], "Key must be 'continuous' or 'discrete'"
        self.exploration[key] = exploration
    
    def get_exploration(self, key: str) -> Exploration:
        """Retrieves exploration config under a named key, e.g. 'continuous' or 'discrete'."""
        return self.exploration.get(key, None)
    
    def del_exploration(self, key: str):
        """Deletes exploration config under a named key, e.g. 'continuous' or 'discrete'."""
        if key in self.exploration:
            del self.exploration[key]

    def list_exploration_keys(self) -> list[str]:
        """
        Returns a list of all keys for which explorations are defined, e.g. ["continuous", "discrete"].
        """
        return list(self.exploration.keys())

    def set_env_host(self, key: str, env_host: EnvHost):
        """Sets a new environment host (endpoint + agent count) in the env_hosts dict."""
        self.env_hosts[key] = env_host
    
    def get_env_host(self, key: str) -> EnvHost:
        """Retrieves an environment host (endpoint + agent count) from the env_hosts dict."""
        return self.env_hosts.get(key, None)
    
    def del_env_host(self, key: str):
        """Deletes an environment host (endpoint + agent count) from the env_hosts dict."""
        if key in self.env_hosts:
            del self.env_hosts[key]

    def list_env_host_keys(self) -> list[str]:
        """
        Returns a list of all environment host keys, e.g. ["local", "remote"].
        """
        return list(self.env_hosts.keys())

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if k == "env_hosts":
                if not isinstance(v, dict):
                    raise TypeError(f"env_hosts must be a dict, got {type(v)}")
                new_dict = {}
                for subkey, item in v.items():
                    if isinstance(item, EnvHost):
                        new_dict[subkey] = item
                    elif isinstance(item, dict):
                        new_dict[subkey] = EnvHost(**item)
                    else:
                        raise TypeError(
                            f"env_hosts values must be a dict or EnvHost, got {type(item)}"
                        )
                self.env_hosts = new_dict

            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in Hyperparameters")

    def to_dict(self) -> dict:
        """
        Returns a dictionary of all fields (useful for logging, JSON export, etc.).
        """
        return vars(self)
    

def main():
    hyperparams = Hyperparameters()

    # Option 1: Use set_config with a dictionary of dicts/EnvHost
    hyperparams.set_config(
        env_hosts={
            "local": {"endpoint": "http://localhost:8000", "num_agents": 64},
            "remote": EnvHost(
                host_id="aws-1",
                env_endpoint="http://ec2-xxx.compute.amazonaws.com",
                num_agents=128
            ),
        },
        env_id="Walker-v2"
    )

    # Option 2: Use set_env_host for single entries
    hyperparams.set_env_host("backup", EnvHost(env_endpoint="http://backup-env.com", num_agents=32))

    # Example of setting exploration
    hyperparams.set_exploration("continuous", Exploration(type="ornstein_uhlenbeck", mu=0.1, theta=0.2))

    # Check results
    config_dict = hyperparams.to_dict()
    print(config_dict["env_hosts"])   # => {'local': EnvHost(...), 'remote': EnvHost(...), 'backup': EnvHost(...)}
    print(config_dict["exploration"]) # => {'continuous': Exploration(...)}
    print("gpt_type =", config_dict["gpt_type"])  # => "gpt2"

if __name__ == "__main__":
    main()