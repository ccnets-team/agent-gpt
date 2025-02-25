from threading import Thread, Event
from .api import EnvAPI

class LocalEnv(EnvAPI):
    """
    LocalEnv extends EnvAPI to manage environment hosting locally.
    It integrates the launching functionality so that you can simply call
    LocalEnv.launch(...) to start a server.
    """
    def __init__(self, env: str, host: str = "0.0.0.0", port: int = 8000):
        if env == 'unity':
            from ..wrappers.unity_env import UnityEnv  # Interface for Unity environments
            env_cls = UnityEnv
        elif env == 'gym':
            from ..wrappers.gym_env import GymEnv      # Interface for Gym environments
            env_cls = GymEnv
        else:
            raise ValueError("Unknown environment simulator. Choose 'unity' or 'gym'.")
        super().__init__(env_cls, host, port)
        self.public_ip = None
        self.host = host
        self.port = port
        # Create a shutdown event that can be used for graceful termination.
        self.shutdown_event = Event()

    def run_thread_server(self):
        """Run the server in a separate daemon thread with a graceful shutdown mechanism."""
        # Note: daemon=True means the thread won't block the process from exiting,
        # but we are joining on it to block the main thread until termination.
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        return self.server_thread

    def shutdown(self):
        """Signal the server to shut down gracefully."""
        self.shutdown_event.set()
        # Additional logic would be needed here to stop the uvicorn server, etc.

    @classmethod
    def launch(cls, env: str, ip: str, host: str = "0.0.0.0", port: int = 8000) -> "LocalEnv":
        """
        Create a LocalEnv instance, launch its server in a separate thread,
        and set the public URL (defaulting to http://host:port).
        """
        instance = cls(env, host, port)
        print(f"[AgentGPTTrainer] Launching local environment at http://{ip}:{port}")
        instance.run_thread_server()
        # Here you can implement logic to auto-detect external IP if needed.
        instance.public_ip = f"http://{ip}:{port}"
        return instance
