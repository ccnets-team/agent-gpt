import argparse
import typer
import os
import platform
import subprocess
from typing import List, Dict

def open_simulation_in_screen(extra_args: List[str]) -> subprocess.Popen:
    env = os.environ.copy()
    simulation_script = os.path.join(os.path.dirname(__file__), "simulation.py")
    system = platform.system()

    if system == "Linux":
        cmd_parts = ["python3", simulation_script] + extra_args
        if not os.environ.get("DISPLAY"):
            # Headless mode: run in background without opening a terminal emulator.
            proc = subprocess.Popen(cmd_parts, env=env)
        else:
            cmd_str = " ".join(cmd_parts)
            try:
                proc = subprocess.Popen(
                    ['gnome-terminal', '--', 'bash', '-c', f'{cmd_str}; exec bash'],
                    env=env
                )
            except FileNotFoundError:
                proc = subprocess.Popen(
                    ['xterm', '-e', f'{cmd_str}; bash'],
                    env=env
                )
    elif system == "Darwin":
        cmd_parts = ["python3", simulation_script] + extra_args
        cmd_str = " ".join(cmd_parts)
        apple_script = (
            'tell application "Terminal"\n'
            f'  do script "{cmd_str}"\n'
            '  activate\n'
            'end tell'
        )
        proc = subprocess.Popen(['osascript', '-e', apple_script], env=env)
    elif system == "Windows":
        cmd_parts = ["python", simulation_script] + extra_args
        cmd_str = " ".join(cmd_parts)
        cmd = f'start cmd /k "{cmd_str}"'
        proc = subprocess.Popen(cmd, shell=True, env=env)
    else:
        typer.echo("Unsupported OS for launching a new terminal session.")
        raise typer.Exit(code=1)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection_identifier", required=True)
    parser.add_argument("--env_type", required=True)
    parser.add_argument("--num_agents", type=int, required=True)
    args = parser.parse_args()

    from agent_gpt.utils.config_utils import load_config, save_config
    from agent_gpt.env_host.server import EnvServer
    from agent_gpt.config.hyperparams import EnvHost
    import typer

    config_data = load_config()

    connection_identifier = args.connection_identifier
    num_launchers = 4

    launchers = [EnvServer.launch(args.env_type) for _ in range(num_launchers)]

    base_agents, remainder = divmod(args.num_agents, num_launchers)
    agents_per_launcher = [base_agents + (1 if i < remainder else 0) for i in range(num_launchers)]

    env_host = config_data.get("hyperparams", {}).get("env_host", {})
    added_env_hosts = []
    
    for i, launcher in enumerate(launchers):
        key = f"local{i + 1}"
        env_host[key] = {"env_endpoint": launcher.connection_key, "num_agents": agents_per_launcher[i]}
        added_env_hosts.append(key)
        
    config_data["hyperparams"]["connection_identifier"] = connection_identifier  # fixed plural naming consistency
    config_data["hyperparams"]["env_host"] = env_host
    save_config(config_data)

    typer.echo("Simulation running. This terminal is now dedicated to simulation;")
    typer.echo("Press Ctrl+C to terminate the simulation.")

    try:
        while any(launcher.server_thread.is_alive() for launcher in launchers):
            for launcher in launchers:
                launcher.server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        typer.echo("Shutdown requested, stopping all servers...")
        for launcher in launchers:
            launcher.shutdown()
        for launcher in launchers:
            launcher.server_thread.join(timeout=2)

    # Cleanup after simulation ends
    config_data = load_config()

    for key in added_env_hosts:
        env_host.pop(key, None)
        
    config_data["hyperparams"]["env_host"] = env_host
    if "connection_identifier" in config_data["hyperparams"]:
        del config_data["hyperparams"]["connection_identifier"]
        
    save_config(config_data)

    typer.echo("Simulation terminated.")

if __name__ == "__main__":
    main()
