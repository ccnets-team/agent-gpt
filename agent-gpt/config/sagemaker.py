from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class TrainerConfig:
    image_uri: str = "533267316703.dkr.ecr.ap-northeast-2.amazonaws.com/agent-gpt-trainer:latest"
    output_path: Optional[str] = "s3://your-bucket/output/"
    instance_type: str = "ml.g5.4xlarge"
    instance_count: int = 1
    max_run: int = 3600

@dataclass
class InferenceConfig:
    image_uri: str = "533267316703.dkr.ecr.ap-northeast-2.amazonaws.com/agent-gpt-inference:latest"
    model_data: Optional[str] = "s3://your-bucket/model.tar.gz"
    endpoint_name: Optional[str] = "agent-gpt-inference-endpoint"
    instance_type: str = "ml.g5.4xlarge"
    instance_count: int = 1
    max_run: int = 3600

@dataclass
class SageMakerConfig:
    role_arn: Optional[str] = "arn:aws:iam::<your-account-id>:role/SageMakerExecutionRole"
    region: Optional[str] = "ap-northeast-2"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def to_dict(self) -> dict:
        """Returns a nested dictionary of the full SageMaker configuration."""
        return asdict(self)
