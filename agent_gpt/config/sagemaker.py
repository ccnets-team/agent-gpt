from dataclasses import dataclass, field, asdict
from typing import Optional
import re

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
    instance_type: str = "ml.t2.medium"
    instance_count: int = 1
    max_run: int = 3600

@dataclass
class SageMakerConfig:
    role_arn: Optional[str] = "arn:aws:iam::<your-aws-account-id>:role/SageMakerExecutionRole"
    region: Optional[str] = "ap-northeast-2"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        # Convert nested dictionaries to their respective dataclass instances if needed.
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)
        if isinstance(self.inference, dict):
            self.inference = InferenceConfig(**self.inference)
    
    def set_region(self, region: str) -> None:
        allowed_regions = ["us-east-1", "ap-northeast-2"]  # ap-northeast-2 corresponds to Seoul
        if region not in allowed_regions:
            raise ValueError(f"Region {region} is not allowed. Allowed regions: {allowed_regions}")
        
        self.region = region
        # This regex pattern matches the region between 'dkr.ecr.' and '.amazonaws.com'
        pattern = r"(dkr\.ecr\.)([\w-]+)(\.amazonaws\.com)"
        
        self.trainer.image_uri = re.sub(pattern, rf"\1{region}\3", self.trainer.image_uri)
        self.inference.image_uri = re.sub(pattern, rf"\1{region}\3", self.inference.image_uri)

    def to_dict(self) -> dict:
        """Returns a nested dictionary of the full SageMaker configuration."""
        return asdict(self)
    
    def set_config(self, **kwargs):
        """
        Update the SageMakerConfig instance using provided keyword arguments.
        For nested fields like 'trainer' and 'inference', update only the specified sub-attributes.
        """
        for k, v in kwargs.items():
            if k == "trainer" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.trainer, sub_key):
                        setattr(self.trainer, sub_key, sub_value)
                    else:
                        print(f"Warning: TrainerConfig has no attribute '{sub_key}'")
            elif k == "inference" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.inference, sub_key):
                        setattr(self.inference, sub_key, sub_value)
                    else:
                        print(f"Warning: InferenceConfig has no attribute '{sub_key}'")
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in SageMakerConfig")