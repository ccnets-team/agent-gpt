# env_hosting/local_host/local_env_launcher.py

from threading import Thread
from src.env_host.api import EnvAPI
from src.env_host.local.tunnel_manager import TunnelManager

class LocalEnvLauncher(EnvAPI):
    """
    LocalEnvLauncher extends EnvironmentAPI to manage environment hosting locally.
    """
    def __init__(self, env_simulator: str, host: str = "0.0.0.0", port: int = 8000):
        if env_simulator == 'unity':
            from wrappers.unity_env import UnityEnv  # Interface for Unity environments
            env_simulator_cls = UnityEnv
        elif env_simulator == 'gym':
            from wrappers.gym_env import GymEnv      # Interface for Gym environments
            env_simulator_cls = GymEnv
        else:
            raise ValueError("Unknown environment simulator. Choose 'unity' or 'gym'.")
        super().__init__(env_simulator_cls, host, port)
        self.tunnel_manager = None
        self.public_url = None
        self.host = host
        self.port = port

    def run_thread_server(self):
        """Run the server in a separate thread."""
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        return self.server_thread

    def run_tunnel(self, tunnel_name: str):
        """Start the tunnel."""
        self.tunnel_manager = TunnelManager(tunnel_name, self.host, self.port)
        self.public_url = self.tunnel_manager.get_public_url()  
        return self.public_url