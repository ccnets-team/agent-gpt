import socket

def get_network_info():
    import requests
    """
    Returns a dictionary with:
    - 'public_ip': The public IP address (if retrievable)
    - 'internal_ip': The local LAN IP address
    """
    info = {
        "public_ip": None,
        "internal_ip": None,
    }

    # 1. Get Public IP via an external service
    try:
        response = requests.get("https://api.ipify.org", timeout=5)
        if response.status_code == 200:
            info["public_ip"] = response.text.strip()
    except requests.RequestException:
        pass  # Handle or log error as needed

    # 2. Get Internal (LAN) IP
    # Using a trick: connect to a known IP (e.g. 8.8.8.8) without sending data
    # just to see what the local address is on that interface.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            info["internal_ip"] = s.getsockname()[0]
    except OSError:
        info["internal_ip"] = "127.0.0.1"  # fallback if unable to detect

    return info

if __name__ == "__main__":
    network_info = get_network_info()
    print(network_info)
