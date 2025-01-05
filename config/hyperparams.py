# conifg/hyperparams.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Hyperparameters:
    """
    A single, consolidated dataclass holding hyperparameters and configurations
    for your RL system (environment, training, optimization, etc.).

    Note: SageMaker-specific fields have been split out into the separate
    SageMakerConfig class.

    Sections
    --------
    1) Client / Env
        - env_id
        - env_endpoint
        - use_tensorboard
        - use_cloudwatch

    2) Session
        - use_print
        - use_graphics
        - max_test_episodes

    3) Training
        - num_agents
        - batch_size
        - train_interval
        - max_steps
        - buffer_size

    4) Algorithm
        - gamma_init
        - lambda_init
        - max_input_states
        - exploration

    5) Optimization
        - lr_init
        - lr_end
        - lr_scheduler
        - lr_cycle_steps
        - tau
        - max_grad_norm

    6) Network
        - num_layers
        - d_model
        - dropout
        - num_heads

    Extra
    -----
    Additional key-value pairs can be placed into extra_hyperparams 
    (e.g., via set_config) if they don’t already have a dedicated field.
    """

    # --------------------
    # 1) Client / Env
    # --------------------
    env_id: Optional[str] = None
    env_endpoint: Optional[str] = None
    use_tensorboard: bool = True
    use_cloudwatch: bool = True
    # use_wandb: bool = False

    # --------------------
    # 2) Session
    # --------------------
    use_print: bool = True
    # resume_train: bool = False
    use_graphics: bool = False
    max_test_episodes: int = 100

    # --------------------
    # 3) Training
    # --------------------
    num_agents: int = 64
    batch_size: int = 64
    train_interval: int = 1
    max_steps: int = 500_000
    buffer_size: int = 500_000

    # --------------------
    # 4) Algorithm
    # --------------------
    gamma_init: float = 0.99
    lambda_init: float = 0.95
    max_input_states: int = 16
    exploration = {
        "type": "gaussian_noise", # "none", "epsilon_greedy", "gaussian_noise", "ornstein_uhlenbeck", "parameter_noise"

        # EpsilonGreedy
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,

        # GaussianNoise
        "initial_sigma": 0.1,
        "final_sigma": 0.001,

        # OrnsteinUhlenbeckNoise
        "mu": 0.0,
        "theta": 0.15,
        "ou_sigma": 0.2, 
        "dt": 1e-2,

        # ParameterNoise
        "initial_stddev": 0.05,
        "final_stddev": 0.0005,
    }
    
    # --------------------
    # 5) Optimization
    # --------------------
    lr_init: float = 1e-4
    lr_end: float = 1e-6
    lr_scheduler: str = "linear"
    lr_cycle_steps: int = 20_000
    tau: float = 0.01
    max_grad_norm: float = 1.0

    # --------------------
    # 6) Network
    # --------------------
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.1
    num_heads: int = 8

    # --------------------
    # Methods
    # --------------------
    
    def set_config(self, **kwargs):
        """
        Dynamically update fields in this dataclass.
        For example: obj.set_config(env_id='Walker-v2', lr_init=1e-5)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in Hyperparameters")
                
    def to_dict(self) -> dict:
        """
        Returns a dictionary of all fields (useful for logging, JSON export, etc.).
        """
        return vars(self)
