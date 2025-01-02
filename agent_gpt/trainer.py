import re
from sagemaker.estimator import Estimator
import importlib
from threading import Thread
from .hyperparams import Hyperparameters
from .sagemaker_config import SageMakerConfig
from env_hosting.env_api import EnvAPI
import uvicorn
from env_hosting.local_host.local_url_provider import LocalURLProvider

class AgentGPTTrainer(EnvAPI):
    """
    A class that extends RemoteTrainer to add SageMaker training functionality.
    It spins up a Flask server for environment management and can launch
    SageMaker training jobs.
    """
    def __init__(self, env_simulator, hyperparameters: Hyperparameters, tunnel_type, **kwargs):
        super().__init__(env_simulator, env_tag = "-remote.ccnets.org")
        
        self.host = kwargs.get('host', "0.0.0.0")
        self.port = kwargs.get('port', 5000)
        self.use_ngrok = kwargs.get('use_ngrok', False)
        self.use_localtunnel = kwargs.get('use_localtunnel', False)
        self.estimator = None
        self.tunnel = None
        
        self.local_url_provider = LocalURLProvider(env_url=hyperparameters.env_url, port=self.port, tunnel_type="none")
        public_url = self.local_url_provider.get_url()
        hyperparameters.env_url = public_url
                
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        print(f"[AgentGPTTrainer] Environment URL: {public_url}")

    def run_server(self):
        # Run Flask on self.host:self.port
        uvicorn.run(self.app, host=self.host, port=self.port)

    def sagemaker_train(self, sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """Launch a SageMaker training job for a one-click robotics environment."""
        # The environment endpoint is whichever public_url we established
        hyperparameters.model_dir = sagemaker_config.model_dir

        self._validate_sagemaker(sagemaker_config)
        self._validate_oneclick(hyperparameters)
        
        if self.env_tag is not None:
            hyperparameters.env_id += self.env_tag # Append a suffix for clarity

        hyperparameters = hyperparameters.to_dict()

        self.estimator = Estimator(
            image_uri=sagemaker_config.trainer_uri,
            role=sagemaker_config.role_arn,
            instance_type=sagemaker_config.instance_type,
            instance_count=sagemaker_config.instance_count,
            output_path=sagemaker_config.model_dir,
            max_run=sagemaker_config.max_run,
            region=sagemaker_config.region,
            hyperparameters=hyperparameters
        )
        self.estimator.fit()

    def _validate_sagemaker(self, sagemaker_config: SageMakerConfig):
        """Validate the SageMaker training job configuration."""
        if (not sagemaker_config.role_arn or 
                not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sagemaker_config.role_arn)):
            print("Role ARN:", sagemaker_config.role_arn)
            raise ValueError("Must provide a valid AWS IAM Role ARN.")

    def _validate_oneclick(self, params: Hyperparameters):
        """Validate the SageMaker training job configuration."""
        if params.env_id is None:
            raise ValueError("Must provide an environment ID.")
        if params.env_url is None:
            raise ValueError("Must provide an environment URL.")
