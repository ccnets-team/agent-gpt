
from remote.agent_manager import AgentManager
from remote.agent_gpt_config import SageMakerConfig
from sagemaker import Model

class AgentGPT(AgentManager):
    def __init__(self, sagemaker_config: SageMakerConfig):
        """
        :param model_path: Where your model is stored (e.g. s3://...).
        :param api_url: The base URL of the remote service (EnvRemoteRunner).
        :param default_env_id: Default environment ID to use.
        """
        model_dir: str = sagemaker_config.model_dir
        api_url: str = sagemaker_config.api_uri

        super().__init__(api_url, model_dir)

        self.model_dir = model_dir
        self.api_url = api_url

        self.inference_model = Model(
            image_uri = self.api_url,
            model_data=self.model_dir,
            role=sagemaker_config.role_arn,
        )
        
        self.predictor = self.inference_model.deploy(
            initial_instance_count=sagemaker_config.instance_count,
            instance_type=sagemaker_config.instance_type
        )
        
        pass