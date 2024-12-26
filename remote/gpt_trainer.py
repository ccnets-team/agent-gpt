# remote_agent_trainer.py
from sagemaker.estimator import Estimator
import re
import logging
from .one_click_params import OneClickHyperparameters, SageMakerConfig
from .remote_env_host import RemoteEnvHost

def training_validator(sage_config: SageMakerConfig, oneclick_params: OneClickHyperparameters):
    """Validate the SageMaker training job configuration."""
    if not sage_config.role_arn or not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sage_config.role_arn):
        raise ValueError("Must provide a valid AWS IAM Role ARN.")
    if oneclick_params.env_id is None:
        raise ValueError("Must provide an environment ID.")
    if oneclick_params.env_url is None:
        raise ValueError("Must provide an environment URL.")

class AgentGPTTrainer(RemoteEnvHost):
    """
    A class that extends RemoteTrainer to add SageMaker training functionality.
    It spins up a Flask server for environment management and can launch
    SageMaker training jobs.
    """
    def __init__(self, env_simulator, port = 5000):
        super().__init__(env_simulator, port)
        self.server_thread.start()
        self.estimator = None
        
    def train(self, sage_config: SageMakerConfig, one_click_params: OneClickHyperparameters):
        """Launch a SageMaker training job for a one-click robotics environment."""
        training_validator(sage_config, one_click_params)

        # Default values from SageMakerConfig + one-click hyperparameters
        hyperparams = {
            "env_id":      one_click_params.env_id,
            "env_url":     one_click_params.env_url,
            "output_path": sage_config.output_path,
        }
        self.estimator = Estimator(
            entry_point='train.py',
            role=sage_config.role_arn,
            instance_type=sage_config.instance_type,
            instance_count=sage_config.instance_count,
            output_path=sage_config.output_path,
            image_uri=sage_config.image_uri,
            max_run=sage_config.max_run,
            hyperparameters=hyperparams
        )

        print("[INFO] Final configuration:")
        for k, v in sage_config.items():
            print(f"  sage_config[{k}] = {v}")
        for k, v in hyperparams.items():
            print(f"  hyperparams[{k}] = {v}")

        # Start the training job
        self.estimator.fit()
        logging.info("SageMaker training job completed (synchronous).")
        
        return self.estimator

    def close(self):
        """
        Close the server and free resources.
        - If training is ongoing (rare in synchronous .fit() usage), you could stop it here.
        - Join the Flask server thread to end the application.
        - Call the parent class's close() to clean up environment(s).
        """

        # If you have an ongoing training job, you could forcibly stop it here:
        # if self.estimator is not None:
        #     try:
        #         self.estimator.stop_training_job()
        #     except Exception as e:
        #         print(f"Unable to stop ongoing training job: {e}")

        self.server_thread.join()
        super().close()
