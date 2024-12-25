from dataclasses import dataclass
from typing import Optional

@dataclass
class OneClickHyperparameters:
    """
    A single, consolidated dataclass holding all hyperparameters and configurations
    for your RL system, including environment, training, and SageMaker parameters.
    
    Feel free to rename fields or reorganize as needed.
    """

    # --------------------
    # 1) Client / Env
    # --------------------
    env_id: Optional[str] = None                # e.g. "Humanoid-v5"
    env_url: Optional[str] = None               # e.g. "http://127.0.0.1:5000"
    use_tensorboard: bool = True
    use_wandb: bool = False
    use_cloudwatch: bool = True

    # --------------------
    # 2) Session
    # --------------------
    device: str = "cuda"
    use_print: bool = True
    resume_train: bool = False
    use_graphics: bool = False
    max_test_episodes: int = 100

    # --------------------
    # 3) Training
    # --------------------
    training_env_id: str = "Humanoid-v5"        # For clarity if you want a separate "training" env_id
    num_agents: int = 128
    batch_size: int = 128
    train_interval: int = 1
    max_steps: int = 1_000_000
    buffer_size: int = 1_000_000

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
    lr_scheduler: str = "cyclic"
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
    # 7) SageMaker
    # --------------------
    role_arn: Optional[str] = None             # e.g. "arn:aws:iam::123456789012:role/SageMakerRole"
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    max_run: int = 3600                        # Max training time in seconds
    image_uri: str = "one-click-server-test:latest"
    output_path: Optional[str] = "s3://one-click-server-test-bucket/output/"
    """
    extra_hyperparams can store additional key-value pairs you want to pass to 
    your train.py or to the RL script but which don't have their own field yet.
    """

    # --------------------
    # Methods
    # --------------------
    def set_config(self, **kwargs):
        """
        Dynamically update fields in this dataclass.
        E.g.: obj.set_config(env_id='Walker-v2', lr_init=1e-5)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in OneClickHyperparameters")

    def to_dict(self) -> dict:
        """
        Return a dictionary of all fields (useful for logging, JSON export, etc.).
        """
        return vars(self)
