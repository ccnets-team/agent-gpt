from threading import Thread, Event
from .api import EnvAPI

class EnvServer(EnvAPI):
    """
    EnvServer extends EnvAPI to manage environment hosting locally.
    It integrates the launching functionality so that you can simply call
    EnvServer.launch(...) to start a server.
    """
    def __init__(self, env_type="gym", env_id=None, entry_point=None, host="0.0.0.0", port=8000):
        if env_type.lower() == "unity":
            from ..wrappers.unity_env import UnityEnv
            env_cls = UnityEnv
            print("[serve.py] Using UnityEnv wrapper.")
        elif env_type.lower() == "gym":
            from ..wrappers.gym_env import GymEnv
            env_cls = GymEnv
            print("[serve.py] Using GymEnv wrapper.")
        else:
            raise ValueError(f"Unknown env type '{env_type}'. Choose 'unity' or 'gym'.")

        # Optionally call the parent's initializer
        super().__init__(env_cls, host, port)

        # Register the environment and create the API instance
        if env_id and entry_point:
            env_cls.register(id=env_id, entry_point=entry_point)
            
        self.api = EnvAPI(env_simulator=env_cls, host=host, port=port)
                
        self.public_ip = None
        self.host = host
        self.port = port

        # Create a shutdown event that can be used for graceful termination.
        self.shutdown_event = Event()

    def run_thread_server(self):
        """Run the server in a separate daemon thread with a graceful shutdown mechanism."""
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        return self.server_thread

    def shutdown(self):
        """Signal the server to shut down gracefully."""
        self.shutdown_event.set()
        # Additional logic would be needed here to stop the uvicorn server, etc.

    @classmethod
    def launch(cls, env_type: str, env_id: str = None, entry_point: str = None, ip: str = None, host: str = "0.0.0.0", port: int = 8000) -> "EnvServer":
        """
        Create an EnvServer instance, launch its server in a separate thread,
        and set the public URL (defaulting to http://host:port).
        """
        instance = cls(env_type, env_id, entry_point, host, port)
        instance.run_thread_server()
        # Default ip to host if not provided
        if ip is None:
            ip = host
        print(f"[AgentGPTTrainer] Launching environment at http://{ip}:{port}")
        instance.public_ip = f"http://{ip}:{port}"
        return instance
