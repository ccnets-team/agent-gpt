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
from typing import Optional, List, Dict

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
    
    def _fields_for_type(self) -> List[str]:
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
    remote_training_key : Optional[str] = None

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
    exploration: Dict[str, Exploration] = field(default_factory=dict)

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
        **kwargs
    ):
        assert key in ["continuous", "discrete"], "Key must be 'continuous' or 'discrete'"
        if key in self.exploration:
            raise KeyError(f"Exploration key '{key}' already exists in hyperparameters.")
        self.exploration[key] = Exploration(type=type, **kwargs)
    
    def del_exploration(self, key: str):
        """Deletes exploration config under a named key, e.g. 'continuous' or 'discrete'."""
        if key in self.exploration:
            del self.exploration[key]
        else:
            raise KeyError(f"Exploration key '{key}' not found in hyperparameters.")

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in Hyperparameters")

    def to_dict(self) -> Dict:
        """
        Returns a deep dictionary of all dataclass fields,
        including nested dataclasses, by leveraging asdict().
        """
        return asdict(self)
    