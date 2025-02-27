import sys
import os
import signal
import typer
from .env_host.server import EnvServer

def main():
    # Signal handler to catch SIGTERM and SIGINT
    def handle_shutdown(signum, frame):
        print(f"Signal {signum} received, initiating shutdown...")
        raise KeyboardInterrupt

    # Register signal handlers for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Use command-line arguments (sys.argv[0] is the script name)
    env = sys.argv[1] if len(sys.argv) > 1 else "gym"
    env_id = sys.argv[2] if len(sys.argv) > 2 else None
    entry_point = sys.argv[3] if len(sys.argv) > 3 else None

    host_arg = os.getenv("HOST", "0.0.0.0")
    # Expect PORTS as a comma-separated list, e.g., "8000,8001,8002"
    ports_str = os.getenv("PORTS") or os.getenv("PORT") or "8000"
    port_args = [int(port.strip()) for port in ports_str.split(",") if port.strip()]

    servers = []
    for port in port_args:
        servers.append(EnvServer.launch(env, env_id, entry_point, host_arg, port))

    try:
        # Block until all server threads are done or until a shutdown signal occurs.
        while any(server.server_thread.is_alive() for server in servers):
            for server in servers:
                server.server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        typer.echo("Shutdown requested, stopping all servers...")
        # Gracefully shut down each server (implement shutdown logic in EnvServer if needed)
        for server in servers:
            try:
                server.shutdown()  # Ensure you implement a shutdown method in EnvServer
            except Exception as e:
                print(f"Error shutting down server on port {server.port}: {e}")
        for server in servers:
            server.server_thread.join(timeout=2)

if __name__ == "__main__":
    main()
