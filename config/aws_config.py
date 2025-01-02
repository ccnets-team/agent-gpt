# agent_gpt/aws_config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class EC2Config:
    """
    A dataclass to encapsulate EC2-related configuration and methods
    to create security groups, authorize SSH, and launch an instance.
    """
    region_name: str = "us-east-1"
    ami_id: str = "ami-08c40ec9ead489470"  # Example Amazon Linux 2023 AMI for us-east-1
    instance_type: str = "t2.micro"
    key_name: Optional[str] = None
    security_group_name: str = "my-ssh-sg"
    security_group_description: str = "Security Group for SSH Tunneling"
    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None

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