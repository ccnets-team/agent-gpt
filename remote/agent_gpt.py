
from sagemaker import Model
from remote.agent_gpt_predictor import AgentGPTPredictor
from remote.agent_gpt_trainer import AgentGPTTrainer
from .gpt_trainer_config import Hyperparameters, SageMakerConfig

class AgentGPT:
    def __init__(self, sagemaker_config: SageMakerConfig):
        """
        :param model_path: Where your model is stored (e.g. s3://...).
        :param api_url: The base URL of the remote service (EnvRemoteRunner).
        :param default_env_id: Default environment ID to use.
        """
        self.model_dir: str = sagemaker_config.model_dir
        self.api_url: str = sagemaker_config.api_uri
    
        self.inference_model = Model(
            image_uri = self.api_url,
            model_data=self.model_dir,
            role=sagemaker_config.role_arn,
        )
        
        self.predictor = self.inference_model.deploy(
            initial_instance_count=sagemaker_config.instance_count,
            instance_type=sagemaker_config.instance_type
        )

    @staticmethod
    def run(cls, sagemaker_config: SageMakerConfig, **kwargs):
        
        agent_gpt: AgentGPT = cls(sagemaker_config)
        
        return AgentGPTPredictor(agent_gpt.predictor)

    @staticmethod
    def train(cls, sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters, env_simulator, **kwargs):
        """Launch a SageMaker training job for a one-click robotics environment."""
        if env_simulator is 'unity':
            from envs.unity_env import UnityEnv        # Interface for Unity environments
            env_simulator_cls = UnityEnv
        elif env_simulator is 'gym':
            from envs.gym_env import GymEnv            # Interface for Gym environments
            env_simulator_cls = GymEnv
        trainer: AgentGPTTrainer = cls(env_simulator_cls, kwargs)
        trainer.sagemaker_train(sagemaker_config, hyperparameters)