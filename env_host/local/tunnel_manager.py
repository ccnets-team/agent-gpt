# env_hosting/local_host/local_env_launcher.py

from env_host.local.tunnel.ngrok import NgrokTunnel
from env_host.local.tunnel.localtunnel import LocalTunnel
from urllib.parse import urlparse
from enum import Enum 

class TunnelType(Enum):
    NGROK = "ngrok"
    LOCALTUNNEL = "localtunnel"
    NONE = "none"

    def __str__(self):
        return self.value

class TunnelManager:
    """
    TunnelManager extends EnvironmentAPI to manage tunneling for local environments
    """
    def __init__(self, tunnel_name, host: str = "0.0.0.0", port: int = 8000):

        self.public_url = None
        self.tunnel = None
        self.host = host
        self.port = port

        # Convert string to enum
        try:
            self.tunnel_name = TunnelType(tunnel_name)
        except ValueError:
            raise ValueError(f"Unknown tunnel_type: {tunnel_name}")

    def _parse_env_url(self, env_endpoint: str):
        """
        If we get an env_endpoint, try to parse out the host/port,
        but only if it's not a known ngrok domain, etc.
        """
        parsed = urlparse(env_endpoint)
        if parsed.hostname and "ngrok" not in parsed.hostname.lower():
            self.host = parsed.hostname
            if parsed.port:
                self.port = parsed.port

    def get_public_url(self) -> str:
        """
        Retrieves a public URL for the environment by either returning 
        the existing URL (if already set) or by creating/opening a tunnel 
        based on the current tunnel_type.

        Returns:
            str: The public URL as a string.
        """
        # 1) If we already have a URL from the constructor or previous call, return it immediately
        if self.public_url:
            return self.public_url

        # 2) Dictionary mapping tunnel types to their corresponding classes
        tunnel_map = {
            TunnelType.LOCALTUNNEL: LocalTunnel,
            TunnelType.NGROK: NgrokTunnel,
        }

        # 3) If tunnel_type is NONE (or None), just use the local URL
        if self.tunnel_name in (TunnelType.NONE, None):
            self.public_url = f"http://{self.host}:{self.port}"
            return self.public_url

        # 4) Otherwise, look up a tunnel class or handle unknown types
        tunnel_cls = tunnel_map.get(self.tunnel_name)
        if tunnel_cls is None:
            # Either raise an error or default to local URL
            # raise ValueError(f"Unsupported tunnel type: {self.tunnel_type}")
            self.public_url = f"http://{self.host}:{self.port}"
            return self.public_url

        # 5) If we have a valid tunnel class, instantiate and open it
        self.tunnel = tunnel_cls(self.host, self.port)
        self.public_url = self.tunnel.open_tunnel()

        return self.public_url