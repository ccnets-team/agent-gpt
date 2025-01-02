# agent_gpt/hyperparams.py
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
        - env_url
        - use_tensorboard
        - use_wandb
        - use_cloudwatch

    2) Session
        - device
        - use_print
        - resume_train
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
        - gpt_seq_len

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
    env_url: Optional[str] = None
    model_dir: Optional[str] = None
    use_tensorboard: bool = True
    # use_wandb: bool = False
    use_cloudwatch: bool = True
    env_tag = "-remote"

    # --------------------
    # 2) Session
    # --------------------
    # device: str = "cuda"
    use_print: bool = True
    # resume_train: bool = False
    use_graphics: bool = False
    max_test_episodes: int = 100

    # --------------------
    # 3) Training
    # --------------------
    num_agents: int = 128
    batch_size: int = 128
    train_interval: int = 1
    max_steps: int = 500_000
    buffer_size: int = 500_000

    # --------------------
    # 4) Algorithm
    # --------------------
    gamma_init: float = 0.99
    lambda_init: float = 0.95
    gpt_seq_len: int = 16

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
    dropout: float = 0.05
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
                print(f"Warning: No attribute '{k}' in OneClickHyperparameters")

    def to_dict(self) -> dict:
        """
        Returns a dictionary of all fields (useful for logging, JSON export, etc.).
        """
        return vars(self)
