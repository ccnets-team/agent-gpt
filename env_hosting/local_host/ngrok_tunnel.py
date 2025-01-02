# ngrok.py
import importlib.util
from pyngrok import ngrok

class NgrokTunnel:
    def __init__(self, port, host, tunnel_config=None):
        self.port = port
        
    def open_tunnel(self):
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