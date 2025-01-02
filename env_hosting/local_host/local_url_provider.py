# local_url_provider.py
from urllib.parse import urlparse

from env_hosting.local_host.ngrok_tunnel import NgrokTunnel
from env_hosting.local_host.localtunnel import LocalTunnel
from env_hosting.local_host.aws_ec2_tunnel import AWSEC2Tunnel

from enum import Enum 

class TunnelType(Enum):
    NGROK = "ngrok"
    AWS_EC2 = "aws_ec2"
    LOCAL_TUNNEL = "localtunnel"
    NONE = "none"

    def __str__(self):
        return self.value
    
class LocalURLProvider:
    """
    LocalURLProvider is responsible for returning a 'public_url'
    that exposes a local service.

    If the user provides an existing env_url, we parse it and
    use that directly. Otherwise, we create a tunnel based on
    the specified tunnel_type (localtunnel, ngrok, aws_ec2, etc.).
    """

    def __init__(
        self,
        port: int = 5000,
        host: str = "127.0.0.1",
        tunnel_type: str = "none",
        tunnel_config = None
    ):
        """
        :param env_url:     If provided, we parse and use it as the public URL.
        :param port:        The local port or the port to expose.
        :param tunnel_type: One of ["localtunnel", "ngrok", "aws_ec2", "none"].
        :param host:        The hostname or IP (default 127.0.0.1).
        :param tunnel_config:  Additional config for the tunnel (e.g., AWS EC2).
        """
        self.port = port
        self.host = host
        self.tunnel_config = tunnel_config
        self.public_url = None
        self.tunnel = None

        # Convert string to enum
        try:
            self.tunnel_type = TunnelType(tunnel_type)
        except ValueError:
            raise ValueError(f"Unknown tunnel_type: {tunnel_type}")

    def _parse_env_url(self, env_url: str):
        """
        If we get an env_url, try to parse out the host/port,
        but only if it's not a known ngrok domain, etc.
        """
        parsed = urlparse(env_url)
        if parsed.hostname and "ngrok" not in parsed.hostname.lower():
            self.host = parsed.hostname
            if parsed.port:
                self.port = parsed.port

    def get_url(self) -> str:
        """
        Initialize the correct tunnel object if needed, then open it
        to get a public URL. If we already have 'public_url' (e.g., from env_url),
        we'll just return that unchanged.

        :return: A final public URL as a string.
        """
        # If we already have an env_url from the constructor, do nothing
        if self.public_url:
            return self.public_url

        if self.tunnel_type == TunnelType.LOCAL_TUNNEL:
            self.tunnel = LocalTunnel(self.port, self.host, self.tunnel_config)
        elif self.tunnel_type == TunnelType.NGROK:
            self.tunnel = NgrokTunnel(self.port, self.host, self.tunnel_config)
        elif self.tunnel_type == TunnelType.AWS_EC2:
            self.tunnel = AWSEC2Tunnel(self.port, self.host, self.tunnel_config)
        elif self.tunnel_type == TunnelType.NONE:
            # No tunnel, just local
            self.public_url = f"http://{self.host}:{self.port}"
            return self.public_url

        # Open the tunnel if we have one
        if self.tunnel:
            self.public_url = self.tunnel.open_tunnel()
        else:
            # Fallback: local URL
            self.public_url = f"http://{self.host}:{self.port}"

        return self.public_url
