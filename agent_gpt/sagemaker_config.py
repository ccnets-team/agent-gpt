# agent_gpt/sagemaker_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SageMakerConfig:
    """
    SageMaker-specific configuration.
    """
    role_arn: Optional[str] = None  # e.g. "arn:aws:iam::123456789012:role/SageMakerRole"
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    max_run: int = 3600               # Max training time in seconds
    trainer_uri: str = "agentgpt-trainer.ccnets.org"
    server_uri: str = "agentgpt.ccnets.org"
    region: Optional[str] = "us-east-1"
    model_dir: Optional[str] = "s3://your-bucket/output/"

    def to_dict(self) -> dict:
        """
        Returns a dictionary of all SageMaker configuration fields.
        """
        return vars(self)