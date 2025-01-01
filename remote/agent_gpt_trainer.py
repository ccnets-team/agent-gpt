# agent_gpt_trainer.py
from sagemaker.estimator import Estimator
import re
from .gpt_trainer_config import Hyperparameters, SageMakerConfig
from .env_host import EnvHost
from threading import Thread
import importlib

class AgentGPTTrainer(EnvHost):
    """
    A class that extends RemoteTrainer to add SageMaker training functionality.
    It spins up a Flask server for environment management and can launch
    SageMaker training jobs.
    """
    def __init__(self, env_simulator, env_id, **kwargs):
        """
        :param env_simulator: Your environment simulator class or instance.
        :param use_ngrok: If True, attempt to tunnel with pyngrok. 
                          If pyngrok is not found, fallback or raise error.
        :param use_ssh: If True, attempt to create an SSH reverse tunnel to my-remote-server.
        :param host: The host/IP to run Flask on. Default 0.0.0.0 for all interfaces.
        :param port: The local port for Flask to listen on. Default 5000.
        """
        super().__init__(env_simulator)
        self.host = kwargs.get('host', '0.0.0.0')
        self.port = kwargs.get('port', 5000)
        self.use_pinggy_tunnel = kwargs.get('use_pinggy_tunnel', False)
        self.use_localtunnel = kwargs.get('use_local_tunnel', False)
        self.use_ngrok_tunnel = kwargs.get('use_ngrok_tunnel', False)
        self.env_id = env_id
        self.estimator = None
        self.public_url = None
        self.server_thread = None
        if self.use_pinggy_tunnel:
            from remote.pinggy_tunnel import PinggyTunnel
            pinggy_tunnel = PinggyTunnel(self.port)
            self.public_url = pinggy_tunnel.open_tunnel()
        if self.use_localtunnel:
            from remote.localtunnel import LocalTunnelApp
            local_tunnel_app = LocalTunnelApp(self.port)
            self.public_url = local_tunnel_app.open_localtunnel()
        elif self.use_ngrok_tunnel:
            self.public_url = self.open_ngrok()
        else:
            self.public_url = f"http://{self.host}:{self.port}"
        print(f"[AgentGPTTrainer] Environment URL: {self.public_url}")
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        
    def run_server(self):
        # 2) Run Flask
        self.app.run(host=self.host, port=self.port)
        
    def open_ngrok(self):
        """
        Attempt to import pyngrok. If successful, create a tunnel and start Flask.
        If pyngrok is not installed, raise an ImportError or fallback to local.
        """
        pyngrok_spec = importlib.util.find_spec("pyngrok")
        if pyngrok_spec is None:
            raise ImportError(
                "pyngrok is not installed. Please run `pip install pyngrok` "
                "or set `use_ngrok=False`."
            )

        from pyngrok import ngrok

        # 1) Create the tunnel
        public_url = ngrok.connect(self.port, "http").public_url
        print(f"[GPTTrainer] ngrok tunnel public URL: {public_url}")
        return public_url

    def sagemaker_train(self, sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters):
        """Launch a SageMaker training job for a one-click robotics environment."""
        # The environment endpoint is whichever self.local_url we established
        hyperparameters.env_url = self.public_url
        hyperparameters.model_dir = sagemaker_config.model_dir
        
        self._validate_sagemaker(sagemaker_config)
        self._validate_oneclick(hyperparameters)

        hyperparameters = hyperparameters.to_dict()
        
        self.estimator = Estimator(
            role=sagemaker_config.role_arn,
            instance_type=sagemaker_config.instance_type,
            instance_count=sagemaker_config.instance_count,
            output_path=sagemaker_config.model_dir,
            image_uri=sagemaker_config.trainer_uri,
            max_run=sagemaker_config.max_run,
            region=sagemaker_config.region,
            hyperparameters=hyperparameters
        )
        
        self.estimator.fit()

    def _validate_sagemaker(self, sagemaker_config: SageMakerConfig):
        """Validate the SageMaker training job configuration."""
        if not sagemaker_config.role_arn or not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sagemaker_config.role_arn):
            print("Role ARN:", sagemaker_config.role_arn)
            raise ValueError("Must provide a valid AWS IAM Role ARN.")

    def _validate_oneclick(self, params: Hyperparameters):
        """Validate the SageMaker training job configuration."""
        if params.env_id is None:
            raise ValueError("Must provide an environment ID.")
        if params.env_url is None:
            raise ValueError("Must provide an environment URL.")
