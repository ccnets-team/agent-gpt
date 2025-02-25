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
    instance_type: str = "ml.t2.medium"
    instance_count: int = 1
    max_run: int = 3600

@dataclass
class SageMakerConfig:
    role_arn: Optional[str] = "arn:aws:iam::<your-account-id>:role/SageMakerExecutionRole"
    region: Optional[str] = "ap-northeast-2"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def __post_init__(self):
        # Convert nested dictionaries to their respective dataclass instances if needed.
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)
        if isinstance(self.inference, dict):
            self.inference = InferenceConfig(**self.inference)
    
    def to_dict(self) -> dict:
        """Returns a nested dictionary of the full SageMaker configuration."""
        return asdict(self)
    
    def set_config(self, **kwargs):
        """
        Updates the SageMakerConfig instance using provided keyword arguments.
        
        For nested fields like 'trainer' and 'inference', the method supports either
        a dict (which is used to construct a new instance) or an already instantiated object.
        """
        for k, v in kwargs.items():
            if k == "trainer":
                if isinstance(v, dict):
                    self.trainer = TrainerConfig(**v)
                elif isinstance(v, TrainerConfig):
                    self.trainer = v
                else:
                    raise TypeError(f"'trainer' must be a dict or TrainerConfig, got {type(v)}")
            elif k == "inference":
                if isinstance(v, dict):
                    self.inference = InferenceConfig(**v)
                elif isinstance(v, InferenceConfig):
                    self.inference = v
                else:
                    raise TypeError(f"'inference' must be a dict or InferenceConfig, got {type(v)}")
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in SageMakerConfig")
