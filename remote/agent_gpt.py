
import logging
from remote.remote_agent_host import RemoteAgentHost
from remote.one_click_params import SageMakerConfig
from sagemaker import Model

class AgentGPT(RemoteAgentHost):
    def __init__(self,
                 model_path: str,
                 api_url: str = "http://oneclickrobotics-aws.ccnets.org"):
        """
        :param model_path: Where your model is stored (e.g. s3://...).
        :param api_url: The base URL of the remote service (EnvRemoteRunner).
        :param default_env_id: Default environment ID to use.
        """
        super().__init__(model_path, api_url)
        
        self.model_path = model_path
        self.api_url = api_url

        logging.basicConfig(level=logging.INFO)

    def run(self, sage_config: SageMakerConfig):
        inference_model = Model(
            image_uri = self.api_url,
            model_data=self.model_path,
            role=sage_config.role_arn,
        )
        predictor = inference_model.deploy(
            initial_instance_count=sage_config.instance_count,
            instance_type=sage_config.instance_type
        )
        return predictor