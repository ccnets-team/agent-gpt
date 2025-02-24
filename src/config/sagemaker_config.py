# src/config/aws_config.py

from dataclasses import dataclass
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