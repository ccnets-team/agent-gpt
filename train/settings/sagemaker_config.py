from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SageMakerConfig:
    """
    Configuration parameters for launching a SageMaker training job.
    Integrates with your RL framework or environment gateway to specify
    key AWS SageMaker settings.

    :param role_arn:        AWS IAM Role ARN used by SageMaker to access resources.
    :param instance_type:   Type of instance (e.g., 'ml.g4dn.xlarge').
    :param instance_count:  Number of instances for distributed training.
    :param max_run:         Maximum allowed training time in seconds.
    :param image_uri:       Docker image URI used to run the training container.
    :param output_path:     S3 path (or local) to store training artifacts.
    :param hyperparams:     Optional dictionary of any additional hyperparameters you want
                            to pass through to the training script (train.py, etc.).
    """
    env_id = None
    env_url = None
    role_arn: Optional[str] = None
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    max_run: int = 3600
    image_uri: str = "one-click-server-test:latest"
    output_path: Optional[str] = "s3://your_bucket/output/"
    hyperparams: dict = field(default_factory=dict)

    def __post_init__(self):
        # Example validation or logging
        if not self.role_arn or not self.env_id or not self.env_url:
            print("Warning: 'role_arn' is not set. SageMaker training may fail without a valid IAM role.")
            print( f"Warning: 'env_id' is not set. SageMaker training may fail without a valid environment ID.")
            print( f"Warning: 'env_url' is not set. SageMaker training may fail without a valid environment URL.")
