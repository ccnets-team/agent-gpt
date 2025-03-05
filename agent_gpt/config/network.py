from dataclasses import dataclass, asdict
import socket
import requests
from typing import Dict

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

    def to_dict(self) -> Dict:
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
                
def get_network_info() -> Dict:
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
    # Determine the internal (LAN) IP address.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            info["internal_ip"] = s.getsockname()[0]
    except OSError:
        info["internal_ip"] = "127.0.0.1"  # Fallback

    try:
        response = requests.get("https://api.ipify.org", timeout=5)
        if response.status_code == 200:
            info["public_ip"] = response.text.strip()
    except requests.RequestException:
        info["public_ip"] = info["internal_ip"]
        print("Warning: Unable to retrieve public IP address.")

    return info

def remote_make(env_key, env_endpoint, 
                env_id, env_entry_point, env_dir,
                timeout=60):
    """
    Sends a POST request to the /make endpoint using msgpack to create an environment.
    
    Args:
        env_key (str): Unique identifier for the environment.
        env_id (str): Environment ID (e.g., "CartPole-v1").
        env_endpoint (str): URL of the remote environment server.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
    
    Returns:
        str: Message returned from the /make endpoint.
    
    Raises:
        Exception: If the HTTP request fails or returns a non-OK status.
    """
    import requests, msgpack

    payload = {
        "env_key": env_key,
        "env_id": env_id,
        "env_entry_point": env_entry_point,
        "env_dir": env_dir,
        "render_mode": None,
    }
    packed = msgpack.packb(payload, use_bin_type=True)
    
    response = requests.post(
        f"{env_endpoint}/make",
        data=packed,
        headers={"Content-Type": "application/x-msgpack"},
        timeout=timeout,
    )
    if not response.ok:
        raise Exception(f"Failed to create env: {response.status_code} - {response.text}")
    
    response_data = msgpack.unpackb(response.content, raw=False)
    make_response = response_data.get("message", "No message returned from /make")
    return make_response

def remote_close(env_key, env_endpoint, timeout=60):
    """
    Sends a POST request to the /close endpoint using msgpack to close an environment.
    
    Args:
        env_key (str): Unique identifier for the environment.
        env_endpoint (str): URL of the remote environment server.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
    
    Returns:
        str: Message returned from the /close endpoint.
    
    Raises:
        Exception: If the HTTP request fails or returns a non-OK status.
    """
    import requests, msgpack

    payload = {"env_key": env_key}
    packed = msgpack.packb(payload, use_bin_type=True)
    
    response = requests.post(
        f"{env_endpoint}/close",
        data=packed,
        headers={"Content-Type": "application/x-msgpack"},
        timeout=timeout,
    )
    if not response.ok:
        raise Exception(f"Failed to close environment {env_key}: {response.status_code} - {response.text}")
    
    response_data = msgpack.unpackb(response.content, raw=False)
    close_message = response_data.get("message", "No message returned from /close")
    return close_message

def test_my_remote_environment(env_hosts: dict, 
                               env_id="Humanoid-v5", env_entry_point="gym.envs:HumanoidEnv", 
                               env_dir="your/env/dir/"):
    """
    Tests remote environment endpoints by creating and then closing an environment.
    
    For each host in the provided dictionary, a unique env_key is generated.
    The function calls `remote_make` to create an environment and then `remote_close` to close it.
    
    Args:
        env_hosts (dict): Dictionary where keys are host identifiers and values contain an "env_endpoint".
        env_id (str, optional): Environment ID to use for testing. Defaults to "remote_dummy_env".
    
    Returns:
        dict: Mapping from each host identifier to a result dictionary with the responses or error message.
    """
    import uuid
    results = {}
    for env_host_id, env_host_data in env_hosts.items():
        env_endpoint = env_host_data.get("env_endpoint")
        if not env_endpoint:
            results[env_host_id] = "No endpoint URL provided."
            continue
        
        env_key = uuid.uuid4().hex[:16]
        try:
            make_response = remote_make(env_key, env_endpoint, env_id, env_entry_point, env_dir)
            close_response = remote_close(env_key, env_endpoint)
            results[env_host_id] = {
                "status": 200,
                "remote-gym-make": make_response,
                "remote-gym-close": close_response
            }
        except Exception as e:
            results[env_host_id] = {
                "status": 500,
                "error": str(e)
            }
    return results

if __name__ == "__main__":
    config = NetworkConfig.from_network_info()
    print(config.to_dict())
