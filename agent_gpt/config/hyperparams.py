"""
AgentGPT Trainer Hyperparameters

We train a contextual (transformer-based) model for action prediction 
using CCNets-based GPT architecture.

CCNets: https://www.linkedin.com/company/ccnets/
Invertible-reasoning policy and reverse dynamics for causal reinforcement learning:
https://patents.google.com/patent/WO2023167576A2/en 

The parameters below are standard RL settings plus additional GPT/transformer fields.

Quick-Reference for Key Fields:
-------------------------------------------------------------------
env_id               : (str)    The name or identifier of your environment 
                                (e.g. 'CartPole-v1', 'Walker-v2', or custom).

env_hosts            : (dict)   A dictionary of EnvHost objects (keyed by
                                strings like 'local', 'remote-aws', etc.) 
                                that define parallel environment endpoints. 
                                Each EnvHost contains an endpoint (URL/IP) 
                                and a number of agents to run there. This
                                mechanism allows distributed or local/cloud-hosted 
                                environments (at different URLs) to send
                                experiences to a single trainer.

replay_ratio         : (float)  Ratio of training iterations to environment steps.
                                For example, 1.0 => one training iteration 
                                per environment step. It is a target to control “backpressure” 
                                when using slow or remote envs.
                                
gamma_init, lambda_init :       Common RL discount factor (gamma) and lambda 
                                used in advantage estimation. Here, both gamma 
                                and lambda are treated as adjustable (learnable) 
                                parameters to help with advantage normalization. 
                                Typically near 1.0 for longer-horizon tasks.

max_input_states     : (int)    Sequence length / context window size for GPT-based 
                                model. Larger values let the model see more context 
                                per inference step, but can increase memory costs.
                                
tau                  : (float)  “Soft update” factor for target networks.
                                For example, 0.01 => slow, more stable updates.

max_grad_norm        : (float)  Gradient clipping threshold. 
                                Lower => more stable training.

gpt_type             : (str)    GPT variant (e.g., "gpt2", "gpt-neo", etc.) from
                                the Hugging Face Transformers library.
-------------------------------------------------------------------
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class EnvHost:
    """
    Holds the env_endpoint info and agent count for a single hosting environment 
    (whether it's local or remote).
    """
    env_endpoint: str = ''                  # e.g., "http://localhost:8000" or "http://ec2-xxx.compute.amazonaws.com"
    num_agents: int = 128

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

    # 1) Client / Env
    env_id: Optional[str] = None
    env_hosts: dict[str, EnvHost] = field(default_factory=dict)
    use_tensorboard: bool = False
    use_cloudwatch: bool = True

    # 2) Session
    use_graphics: bool = False
    resume_training: bool = False       # If True, trainer loads from checkpoint/optimal. fill in checkpoint path
    
    # 3) Training
    batch_size: int = 256
    replay_ratio: float = 2.0
    max_steps: int = 20_000_000
    buffer_size: int = 1_000_000

    # 4) Algorithm
    gamma_init: float = 0.99
    lambda_init: float = 0.95
    max_input_states: int = 16
    exploration: dict[str, Exploration] = field(default_factory=dict)

    # 5) Optimization
    lr_init: float = 1e-4
    lr_end: float = 1e-5
    lr_scheduler: str = "linear"  # "linear", "exponential",
    tau: float = 0.01
    max_grad_norm: float = 0.5

    # 6) Network
    gpt_type: str = "gpt2"  
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.1
    num_heads: int = 8
    
    # -----------------------
    # Methods
    # -----------------------
    def set_exploration(
        self,
        key: str,
        type: str = "gaussian_noise",
    ):
        assert key in ["continuous", "discrete"], "Key must be 'continuous' or 'discrete'"
        
        self.exploration[key] = Exploration(type=type)
    
    def del_exploration(self, key: str):
        """Deletes exploration config under a named key, e.g. 'continuous' or 'discrete'."""
        if key in self.exploration:
            del self.exploration[key]

    def set_env_host(self, key: str, env_endpoint: str, num_agents: int):
        """Sets a new environment host (endpoint + agent count) in the env_hosts dict."""
        self.env_hosts[key] = EnvHost(env_endpoint=env_endpoint, num_agents=num_agents) 

    def del_env_host(self, key: str):
        """Deletes an environment host (endpoint + agent count) from the env_hosts dict."""
        if key in self.env_hosts:
            del self.env_hosts[key]
            
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
        Returns a deep dictionary of all dataclass fields,
        including nested dataclasses, by leveraging asdict().
        """
        return asdict(self)
    
def main():
    # 1) Instantiate hyperparameters with defaults
    hyperparams = Hyperparameters()

    # 2) Add a single known local host ("local1")
    hyperparams.set_env_host("local1", "http://23.34.13.132:8500", 32)

    # 3) Specify how many additional local hosts to create, starting from index 2
    num_local_hosts = 4  # e.g., will generate local2, local3, local4, local5

    # Base IP and port for these extra hosts
    base_ip = "http://30.14.22.168:"
    starting_port = 45450

    # 4) Loop to add local2..localN, each with a unique endpoint
    for i in range(2, num_local_hosts + 2):
        host_key = f"local{i}"
        # Construct an endpoint using base_ip plus an incremental port number
        env_endpoint = f"{base_ip}{starting_port + i}"
        # Add the host to hyperparams using the new signature
        hyperparams.set_env_host(host_key, env_endpoint, 32)

    # 5) Configure exploration parameters for continuous action space using the new API
    hyperparams.set_exploration(
        "continuous",
        type="gaussian_noise",
        initial_sigma=0.1,
        final_sigma=0.001
    )

    # 6) Convert to a dictionary for printing or downstream consumption
    config_dict = hyperparams.to_dict()

    # 7) Print environment hosts
    print("Environment hosts:")
    for key, host_info in config_dict["env_hosts"].items():
        print(f"  {key}: endpoint={host_info['env_endpoint']}, agents={host_info['num_agents']}")

    # 8) Print exploration settings and other highlights
    print("\nExploration configs:", config_dict["exploration"])
    print("GPT type:", config_dict["gpt_type"])
    print("\nFull hyperparams dictionary:\n", config_dict)

if __name__ == "__main__":
    main()