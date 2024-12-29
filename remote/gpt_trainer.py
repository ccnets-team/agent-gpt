# remote_agent_trainer.py
from sagemaker.estimator import Estimator
import re
from .agent_gpt_config import Hyperparameters, SageMakerConfig
from .env_host import EnvHost
from threading import Thread
import importlib

class GPTTrainer(EnvHost):
    """
    A class that extends RemoteTrainer to add SageMaker training functionality.
    It spins up a Flask server for environment management and can launch
    SageMaker training jobs.
    """

    def __init__(self, env_simulator, use_ngrok=False, host='0.0.0.0', port=5000):
        """
        :param env_simulator: Your environment simulator class or instance.
        :param use_ngrok: If True, attempt to tunnel with pyngrok. If pyngrok is not found, fallback or raise error.
        :param host: The host/IP to run Flask on. Default 0.0.0.0 for all interfaces.
        :param port: The port for Flask to listen on. Default 5000.
        """
        super().__init__(env_simulator)
        self.use_ngrok = use_ngrok
        self.host = host
        self.port = port
        
        self.estimator = None
        self.public_url = None
        self.server_thread = None

        if self.use_ngrok:
            self._start_ngrok_and_flask()
        else:
            self._start_flask_only()

    def _start_ngrok_and_flask(self):
        """
        Attempt to import pyngrok. If successful, create a tunnel and start Flask.
        If pyngrok is not installed, raise an ImportError or fallback to local.
        """
        # Check if pyngrok is available
        pyngrok_spec = importlib.util.find_spec("pyngrok")
        if pyngrok_spec is None:
            raise ImportError("pyngrok is not installed. Please run `pip install pyngrok` or set `use_ngrok=False`.")
        
        from pyngrok import ngrok  # Import inside method so it's only used if we want ngrok

        def run_server():
            # 1) Create the tunnel
            self.public_url = ngrok.connect(self.port, "http").public_url
            print(f"[GPTTrainer] ngrok tunnel public URL: {self.public_url}")

            # 2) Run Flask
            self.app.run(host=self.host, port=self.port)

        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def _start_flask_only(self):
        """Start Flask without ngrok, for local or LAN access only."""
        def run_server():
            print(f"[GPTTrainer] Running Flask on http://{self.host}:{self.port} (no tunnel)")
            self.app.run(host=self.host, port=self.port)

        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def train(self, sage_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """Launch a SageMaker training job for a one-click robotics environment."""
        
        self._validate_sagemaker(sage_config)
        self._validate_oneclick(hyperparameters)

        # Default values from SageMakerConfig + one-click hyperparameters
        hyperparameters = hyperparameters.to_dict()
        
        sage_config.set_role_arn_from_sts()
        
        self.estimator = Estimator(
            role=sage_config.role_arn,
            instance_type=sage_config.instance_type,
            instance_count=sage_config.instance_count,
            output_path=sage_config.model_dir,
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