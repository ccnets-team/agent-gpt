# File: env_host/serve.py

import sys
import uvicorn
from env_host.api import EnvAPI

def main(simulator="gym", host="0.0.0.0", port=80):
    """
    A simple entrypoint for the Docker container. Creates the EnvAPI with either
    UnityEnv or GymEnv, then runs the server.
    """
    if simulator.lower() == "unity":
        from env_host.wrappers.unity_env import UnityEnv
        env_cls = UnityEnv
        print("[serve.py] Using UnityEnv wrapper.")
    elif simulator.lower() == "gym":
        from env_host.wrappers.gym_env import GymEnv
        env_cls = GymEnv
        print("[serve.py] Using GymEnv wrapper.")
    else:
        raise ValueError(f"Unknown simulator type '{simulator}'. Choose 'unity' or 'gym'.")

    # Create the EnvAPI and run the server
    api = EnvAPI(env_simulator=env_cls, host=host, port=port)
    uvicorn.run(api.app, host=host, port=port)

if __name__ == "__main__":
    # (Optional) parse CLI arguments from sys.argv or environment variables
    # e.g., python serve.py gym 0.0.0.0 80
    simulator_arg = sys.argv[1] if len(sys.argv) > 1 else "gym"
    host_arg = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    port_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 80

    main(simulator_arg, host_arg, port_arg)
