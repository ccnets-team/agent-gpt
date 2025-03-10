import argparse
import typer
import os
import platform
import subprocess
from typing import List

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
    parser.add_argument("--connection_identifiers", required=True)
    parser.add_argument("--env_type", required=True)
    parser.add_argument("--num_agents", type=int, required=True)
    args = parser.parse_args()
    
    from agent_gpt.utils.config_utils import load_config, save_config        
    from agent_gpt.env_host.server import EnvServer    
    
    config_data = load_config()
    
    connection_identifiers = args.connection_identifiers.split(",")  # fixed parsing
    num_launchers = len(connection_identifiers)  # Ensure consistent with identifiers

    launchers = []
    for _ in range(num_launchers):  # fix iteration over range
        launcher = EnvServer.launch(args.env_type)
        launchers.append(launcher)
    
    base_agents = args.num_agents // num_launchers
    remainder = args.num_agents % num_launchers
    agents_array = [base_agents] * num_launchers
    for i in range(remainder):
        agents_array[i] += 1
    
    connection_key = config_data.get("hyperparams", {}).get("connection_key", {})
    added_connection_keys = []
    
    for i, launcher in enumerate(launchers):
        connection_identifier = connection_identifiers[i] 
        connection_key[i] = launcher.connection_key
        added_connection_keys.append(connection_identifier)
        typer.echo(f"Configured connection_key entry: {connection_key} with {agents_array[i]} agents")
    
    config_data.setdefault("hyperparams", {})["connection_key"] = connection_key
    save_config(config_data)
    
    typer.echo("Simulation running. This terminal is now dedicated to simulation;")
    typer.echo("Press Ctrl+C to terminate the simulation.")
    
    try:
        while any(launcher.server_thread.is_alive() for launcher in launchers):
            for launcher in launchers:
                launcher.server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        typer.echo("Shutdown requested, stopping all local servers...")
        for launcher in launchers:
            launcher.shutdown()
        for launcher in launchers:
            launcher.server_thread.join(timeout=2)

    # Cleanup config after simulation ends
    for key in connection_key:
        connection_key.pop(key, None)
        
    config_data["hyperparams"]["connection_key"] = connection_key
    save_config(config_data)
    
    typer.echo("Simulation terminated.")

if __name__ == "__main__":
    main()
