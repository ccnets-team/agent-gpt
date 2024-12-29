# remote_agent_trainer.py
from sagemaker.estimator import Estimator
import re
from .agent_gpt_config import Hyperparameters, SageMakerConfig
from .env_host import EnvHost
from threading import Thread

class GPTTrainer(EnvHost):
    """
    A class that extends RemoteTrainer to add SageMaker training functionality.
    It spins up a Flask server for environment management and can launch
    SageMaker training jobs.
    """
    def __init__(self, env_simulator, host='0.0.0.0', port = 5000):
        super().__init__(env_simulator)
        self.estimator = None
        
        self.server_thread = Thread(target=lambda: self.app.run(host=host, port=port))
        self.server_thread.start()
    
    def train(self, sage_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """Launch a SageMaker training job for a one-click robotics environment."""
        
        self._validate_sagemaker(sage_config)
        self._validate_oneclick(hyperparameters)

        # Default values from SageMakerConfig + one-click hyperparameters
        hyperparameters = hyperparameters.to_dict()
        hyperparameters['output_path'] = sage_config.output_path
        
        hyperparameters.env_id = sage_config.output_path
        
        self.estimator = Estimator(
            role=sage_config.role_arn,
            instance_type=sage_config.instance_type,
            instance_count=sage_config.instance_count,
            output_path=sage_config.output_path,
            image_uri=sage_config.api_uri,
            max_run=sage_config.max_run,
            hyperparameters=hyperparameters
        )
        
        self.estimator.fit()
        pass
    
    # def close(self):
    #     """
    #     Close the server and free resources.
    #     - If training is ongoing (rare in synchronous .fit() usage), you could stop it here.
    #     - Join the Flask server thread to end the application.
    #     - Call the parent class's close() to clean up environment(s).
    #     """

    #     # If you have an ongoing training job, you could forcibly stop it here:
    #     # if self.estimator is not None:
    #     #     try:
    #     #         self.estimator.stop_training_job()
    #     #     except Exception as e:
    #     #         print(f"Unable to stop ongoing training job: {e}")

    #     super().close()
    #     self.server_thread.join()

    def _validate_sagemaker(self, sagemaker_config: SageMakerConfig):
        """Validate the SageMaker training job configuration."""
        if not sagemaker_config.role_arn or not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sagemaker_config.role_arn):
            raise ValueError("Must provide a valid AWS IAM Role ARN.")

    def _validate_oneclick(self, params: Hyperparameters):
        """Validate the SageMaker training job configuration."""
        if params.env_id is None:
            raise ValueError("Must provide an environment ID.")
        if params.env_url is None:
            raise ValueError("Must provide an environment URL.")
    
    @staticmethod
    def sagemaker_train(cls, env_simulator, port = 5000, sage_config: SageMakerConfig= None, hyperparameters: Hyperparameters= None):
        """Launch a SageMaker training job for a one-click robotics environment."""
        
        trainer: GPTTrainer = cls(env_simulator, port)
        trainer.train(sage_config, hyperparameters)