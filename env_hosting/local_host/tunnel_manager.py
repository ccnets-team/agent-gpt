# env_hosting/local_host/tunnel_manager.py
from urllib.parse import urlparse

from env_hosting.local_host.ngrok_tunnel import NgrokTunnel
from env_hosting.local_host.localtunnel import LocalTunnel

from enum import Enum 

class TunnelType(Enum):
    NGROK = "ngrok"
    LOCALTUNNEL = "localtunnel"
    NONE = "none"

    def __str__(self):
        return self.value
    
class TunnelManager:
    """
    TunnelManager is responsible for returning a 'public_url'
    that exposes a local service.

    If the user provides an existing env_endpoint, we parse it and
    use that directly. Otherwise, we create a tunnel based on
    the specified tunnel_type (localtunnel, ngrok, aws_ec2, etc.).
    """

    def __init__(
        self,
        tunnel_name: str = "none",
        host: str = "localhost",
        port: int = 8000
    ):
        """
        :param env_endpoint:     If provided, we parse and use it as the public URL.
        :param port:        The local port or the port to expose.
        :param tunnel_type: One of ["localtunnel", "ngrok", "aws_ec2", "none"].
        :param host:        The hostname or IP (default 0.0.0.0).
        :param tunnel_config:  Additional config for the tunnel (e.g., AWS EC2).
        """
        self.port = port
        self.host = host
        self.public_url = None
        self.tunnel = None

        # Convert string to enum
        try:
            self.tunnel_type = TunnelType(tunnel_name)
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
        if self.tunnel_type in (TunnelType.NONE, None):
            self.public_url = f"http://{self.host}:{self.port}"
            return self.public_url

        # 4) Otherwise, look up a tunnel class or handle unknown types
        tunnel_cls = tunnel_map.get(self.tunnel_type)
        if tunnel_cls is None:
            # Either raise an error or default to local URL
            # raise ValueError(f"Unsupported tunnel type: {self.tunnel_type}")
            self.public_url = f"http://{self.host}:{self.port}"
            return self.public_url

        # 5) If we have a valid tunnel class, instantiate and open it
        self.tunnel = tunnel_cls(self.host, self.port)
        self.public_url = self.tunnel.open_tunnel()

        return self.public_url