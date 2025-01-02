# env_hosting/env_host.py
from dataclasses import dataclass
from typing import Optional
from config.aws_config import EC2Config
from env_hosting.env_api import EnvAPI
import uvicorn
from env_hosting.local_host.local_url_provider import LocalURLProvider
from threading import Thread

class EnvLauncher(EnvAPI):
    def __init__(self, env_simulator):
        if env_simulator == 'unity':
            from env_wrappers.unity_env import UnityEnv        # Interface for Unity environments
            env_simulator_cls = UnityEnv
        elif env_simulator == 'gym':
            from env_wrappers.gym_env import GymEnv            # Interface for Gym environments
            env_simulator_cls = GymEnv
        super().__init__(env_simulator_cls, env_tag = "-remote.ccnets.org")
        
    def run_server(self):
        # Run Flask on self.host:self.port
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def host_on_cloud(self, ec2_config: EC2Config):
        """Set up or reference a cloud environment, return the URL."""
        some_url = None
        return some_url
    
    def host_on_local(self, tunnel_config, host: str = "127.0.0.1", port: int = 8000):
        """Run locally and create a tunnel (Ngrok, LocalTunnel), return the public URL."""
        self.host = host or self.host
        self.port = port or self.port

        self.local_url_provider = LocalURLProvider(tunnel_config)
        tunnel_url = self.local_url_provider.get_url()
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        print(f"[AgentGPTTrainer] Environment URL: {tunnel_url}")
        return tunnel_url
    
    def host_with_url(self, url: str, host: str = "127.0.0.1", port: int = 8000):
        self.host = host or self.host
        self.port = port or self.port
        
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        print(f"[AgentGPTTrainer] Environment URL: {url}")
        return url
