
    
class TunnelConfig:
    def __init__(self, port: int, host: str, tunnel_type: TunnelType):
        self.port = port
        self.host = host
        self.tunnel_type = tunnel_type