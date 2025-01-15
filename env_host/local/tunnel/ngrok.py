# env_hosting/local_host/ngrok_tunnel.py
import importlib.util

class NgrokTunnel:
    def __init__(self, host="localhost", port=8000):
        self.port = port
        
    def open_tunnel(self):
        from pyngrok import ngrok
        """
        Attempt to import pyngrok. If successful, create a tunnel.
        If pyngrok is not installed, raise an ImportError.
        """
        pyngrok_spec = importlib.util.find_spec("pyngrok")
        if pyngrok_spec is None:
            raise ImportError(
                "pyngrok is not installed. Please run `pip install pyngrok` "
                "or set `use_ngrok=False`."
            )

        public_url = ngrok.connect(self.port, "http").public_url
        print(f"[GPTTrainer] ngrok tunnel public URL: {public_url}")
        return public_url
    
# ------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    tunnel = NgrokTunnel(host="localhost", port=8000)
    public_url = tunnel.open_tunnel()
    if public_url:
        print(f"Public URL: {public_url}")
    else:
        print("Failed to create ngrok tunnel.")
# ------------------------------------------------------------------------------
    
    