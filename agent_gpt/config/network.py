from dataclasses import dataclass, asdict
import socket
import requests

@dataclass
class NetworkConfig:
    host: str = "0.0.0.0"  # Defaults to 0.0.0.0; can be overridden by public_ip if available
    public_ip: str = ""
    internal_ip: str = ""
    
    def __post_init__(self):
        # Fetch network info and update fields accordingly.
        info = get_network_info()  # Assumes get_network_info() is defined elsewhere.
        self.public_ip = info.get("public_ip", self.public_ip)
        self.internal_ip = info.get("internal_ip", self.internal_ip) or "127.0.0.1"

    def to_dict(self) -> dict:
        """Returns a dictionary of all Network configuration fields."""
        return asdict(self)

    def set_config(self, **kwargs):
        """
        Updates the NetworkConfig instance based on provided keyword arguments.
        Only updates existing attributes; warns if an unknown key is provided.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in NetworkConfig")
                
    @classmethod
    def from_network_info(cls) -> "NetworkConfig":
        """
        Convenience method that creates an instance.
        __post_init__ automatically fetches network info.
        """
        return cls()
    
                
def get_network_info() -> dict:
    """
    Returns a dictionary with:
    - 'public_ip': The public IP address (if retrievable)
    - 'internal_ip': The local LAN IP address
    """
    info = {
        "public_ip": None,
        "internal_ip": None,
    }
    # Try to get the public IP via an external service.
    try:
        response = requests.get("https://api.ipify.org", timeout=5)
        if response.status_code == 200:
            info["public_ip"] = response.text.strip()
    except requests.RequestException:
        pass  # Ignore errors

    # Determine the internal (LAN) IP address.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            info["internal_ip"] = s.getsockname()[0]
    except OSError:
        info["internal_ip"] = "127.0.0.1"  # Fallback

    return info

if __name__ == "__main__":
    config = NetworkConfig.from_network_info()
    print(config.to_dict())
