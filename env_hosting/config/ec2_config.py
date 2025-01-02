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
