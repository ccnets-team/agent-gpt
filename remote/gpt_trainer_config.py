from dataclasses import dataclass
from typing import Optional
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class SageMakerConfig:
    """
    SageMaker-specific configuration.
    """
    role_arn: Optional[str] = None  # e.g. "arn:aws:iam::123456789012:role/SageMakerRole"
    instance_type: str = None
    instance_count: int = None
    max_run: int = None               # Max training time in seconds
    trainer_uri: str = None
    server_uri: str = None
    region: Optional[str] = None
    model_dir: Optional[str] = None

    def __init__(
        self,
        role_arn: Optional[str] = None,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        max_run: int = 3600,
        trainer_uri: str = "agentgpt-trainer.ccnets.org",
        server_uri: str = "agentgpt.ccnets.org",
        region: Optional[str] = "us-east-1",
        model_dir: Optional[str] = "s3://your-bucket/output/"
    ):
        # Manually assign fields
        self.role_arn = role_arn or self.get_role_arn()
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.max_run = max_run
        self.trainer_uri = trainer_uri
        self.server_uri = server_uri
        self.model_dir = model_dir
        self.region = region

    def get_role_arn(self):
        """
        Retrieve the current caller identity from AWS STS and set
        this dataclass's role_arn to the returned ARN.
        Also prints out the account ID, user ID, and ARN for reference.
        """
        role_arn = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
        if role_arn:
            self.role_arn = role_arn
            print(f"Role ARN: {role_arn}")
            return role_arn
        return None

    def to_dict(self) -> dict:
        """
        Returns a dictionary of all SageMaker configuration fields.
        """
        return dict(
            role_arn=self.role_arn,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            max_run=self.max_run,
            image_uri=self.image_uri,
            model_dir=self.model_dir,
        )

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
