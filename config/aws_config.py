# agent_gpt/aws_config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class EC2Config:
    """
    Holds basic AWS EC2 configuration details.
    Feel free to extend with more fields as needed (e.g., user_data, IAM roles).
    """
    region_name: str = None    # Provide defaults if you like
    ami_id: str = None      # Some base AMI
    instance_type: str = None   # Default instance type
    key_name: Optional[str] = None
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None
    instance_name: Optional[str] = None
    def __init__(self, ami_id: str = None, instance_type: str = "g5.xlarge", instance_name: str = None, region_name: str = "us-east-1", 
                 key_name: str = None, subnet_id: str = None, security_group_id: str = None, ensure_ami_config: bool = False):
        self.ami_id = ami_id
        self.instance_type = instance_type
        self.instance_name = instance_name
        self.region_name = region_name
        self.key_name = key_name
        self.subnet_id = subnet_id
        self.security_group_id = security_group_id
        if self.ami_id is None and ensure_ami_config:
            self.ami_id = self.configure_ami()
        
    def configure_ami(self) -> None:
        """
        Automatically finds the latest Amazon Linux 2 AMI in the given region
        and updates self.ami_id accordingly.
        """
        import boto3
        ec2 = boto3.client("ec2", region_name=self.region_name)

        response = ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": ["amzn2-ami-hvm-*"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "virtualization-type", "Values": ["hvm"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        ami_id = None

        # Sort images by CreationDate descending, so the newest is first
        images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
        if images:
            latest = images[0]
            ami_id = latest["ImageId"]
            
        return ami_id
  
    def to_dict(self) -> dict:
        """
        Returns a dictionary of all SageMaker configuration fields.
        """
        return vars(self)
      
@dataclass
class SageMakerConfig:
    """
    SageMaker-specific configuration.
    """
    role_arn: Optional[str] = None  # e.g. "arn:aws:iam::123456789012:role/SageMakerRole"
    # instance_type: str = "ml.g4dn.xlarge"
    image_uri: str = "agentgpt.ccnets.org"
    output_path: Optional[str] = "s3://your-bucket/output/"
    model_data: Optional[str] = "s3://your-bucket/model.tar.gz"
    instance_type: str = "ml.t2.medium" # for endpoint instance (e.g. ml.t2.medium eq) or training instance (e.g. ml.g4dn.xlarge eq) for trainer instance
    instance_count: int = 1
    region: Optional[str] = "us-east-1"
    max_run: int = 3600               # Max training time in seconds

    def __init__(self, role_arn: str = None, image_uri: str = None, output_path: str = None, model_data: str = None,
                    instance_type: str = "ml.g4dn.xlarge", instance_count: int = 1, region: str = "us-east-1", max_run: int = 3600):
        self.role_arn = role_arn
        self.image_uri = image_uri
        self.output_path = output_path
        self.model_data = model_data
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.region = region
        self.max_run = max_run

    def to_dict(self) -> dict:
        """
        Returns a dictionary of all SageMaker configuration fields.
        """
        return vars(self)