# env_launcher.py

from src.config.aws_config import EC2Config
from src.env_host.local.local_launcher import LocalEnvLauncher
from src.env_host.cloud.cloud_launcher import CloudEnvLauncher

class EnvLauncher:
    """
    EnvLauncher extends EnvironmentAPI to manage environment hosting locally or on the cloud.
    """
    def __init__(self):
        pass

    @staticmethod
    def launch_on_local_with_url(env_simulator:str, tunnel_name: str=None, host: str = "0.0.0.0", port: int = 8000) -> str:
        """
        Runs the server locally and creates a tunnel (e.g., Ngrok, LocalTunnel).
        Returns the public URL for accessing this environment remotely.
        """
        local_env_launcher = LocalEnvLauncher(env_simulator, tunnel_name, host, port)
        tunnel_url = local_env_launcher.run_tunnel(tunnel_name)
        print(f"[AgentGPTTrainer] Environment URL: {tunnel_url}")
        
        local_env_launcher.run_thread_server()
        return local_env_launcher

    @staticmethod
    def launch_on_local_with_ip(env_simulator:str, ip_address: str = None, host: str = "0.0.0.0", port: int = 8000) -> str:
        """
        Runs the server locally, using the provided ip_address (parsing for IP/port if present).
        Returns the environment endpoint used.
        """
        print(f"[AgentGPTTrainer] Environment Endpoint: http://{ip_address}:{port}")
        local_env_launcher = LocalEnvLauncher(env_simulator, host, port)
        local_env_launcher.run_thread_server()
        return local_env_launcher
    
    @staticmethod
    def launch_on_cloud(env_simulator:str, env_id, env_file_path, global_image_name) -> CloudEnvLauncher:
        """
        A static method returning a `CloudEnvLauncher` instance with default config,
        so users can do:

            launcher = CloudEnvLauncher.host_on_cloud()
            # Then call methods like generate_docker_file(), build_docker_image(), etc.
        """
        return CloudEnvLauncher(env_simulator, env_id, env_file_path, global_image_name)
