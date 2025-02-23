# src/agent_gpt/aws_config.py

from dataclasses import dataclass, InitVar
from typing import Optional
      
@dataclass
class SageMakerConfig:
    """
    SageMaker-specific configuration.

    Attributes:
        role_arn: AWS IAM Role ARN used for SageMaker execution (e.g., "arn:aws:iam::123456789012:role/SageMakerRole").
        image_uri: Container image URI for the inference container.
        output_path: S3 path for storing training outputs.
        model_data: S3 path to the model tarball.
        instance_type: Instance type for training or inference.
        instance_count: Number of instances to use.
        region: AWS region for deployment.
        endpoint_name: Name of the SageMaker real-time inference endpoint.
            This endpoint is used for online predictions. If not provided, a default name will be auto-generated.
        max_run: Maximum training runtime in seconds.
    """
    role_arn: Optional[str] = None
    image_uri: str = "agentgpt.ccnets.org"
    output_path: Optional[str] = "s3://your-bucket/output/"
    model_data: Optional[str] = "s3://your-bucket/model.tar.gz"
    instance_type: str = "ml.g5.4xlarge"
    instance_count: int = 1
    region: Optional[str] = "ap-northeast-2"
    endpoint_name: Optional[str] = "agent-gpt-inference-endpoint"
    max_run: int = 3600

    def to_dict(self) -> dict:
        """Returns a dictionary of all SageMaker configuration fields."""
        return vars(self)

@dataclass
class EC2Config:
    """
    Holds basic AWS EC2 configuration details.
    Feel free to extend with more fields as needed (e.g., user_data, IAM roles).
    
    Attributes:
        region_name: AWS region (default "ap-northeast-2" but can be changed to automatic cross region solution)
        ami_id: AMI ID to use; if None and ensure_ami_config is True, the latest Amazon Linux 2 AMI is auto-configured.
        instance_type: Type of EC2 instance (default "ml.g5.xlarge").
        key_name: Name of the EC2 key pair.
        subnet_id: ID of the subnet for the instance.
        security_group_id: Security group ID.
        instance_name: Name for the EC2 instance.
    """
    region_name: str = "ap-northeast-2"
    ami_id: Optional[str] = None
    instance_type: str = "ml.g5.xlarge"
    key_name: Optional[str] = None
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None
    instance_name: Optional[str] = None
    # Using InitVar to allow a parameter that is not stored as a field
    ensure_ami_config: InitVar[bool] = False

    def __post_init__(self, ensure_ami_config: bool):
        if self.ami_id is None and ensure_ami_config:
            self.ami_id = self.configure_ami()

    def configure_ami(self) -> Optional[str]:
        """
        Automatically finds the latest Amazon Linux 2 AMI in the given region
        and returns its AMI ID.
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
        # Sort images by CreationDate descending, so the newest is first
        images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
        if images:
            return images[0]["ImageId"]
        return None

    def to_dict(self) -> dict:
        """
        Returns a dictionary of all EC2 configuration fields.
        """
        return vars(self)