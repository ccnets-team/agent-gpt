import argparse
import time
import typer
import os
import platform
import subprocess

def open_simulation_in_screen(extra_args: list[str]):
    """
    Launch a new terminal/screen session that runs the simulation process.
    extra_args should contain command-line arguments to pass to simulation.py.
    """
    env = os.environ.copy()
    simulation_script = os.path.join(os.path.dirname(__file__), "simulation.py")
    system = platform.system()

    if system == "Linux":
        # Use GNU screen to launch a detached session named "simulation"
        subprocess.Popen(['screen', '-dmS', 'simulation', 'python3', simulation_script] + extra_args, env=env)
    elif system == "Darwin":
        # Use AppleScript to open a new Terminal window on macOS.
        arg_str = " ".join(extra_args)
        apple_script = (
            'tell application "Terminal"\n'
            f'  do script "python3 {simulation_script} {arg_str}"\n'
            '  activate\n'
            'end tell'
        )
        subprocess.Popen(['osascript', '-e', apple_script], env=env)
    elif system == "Windows":
        # Check if running in bash (like Git Bash) by checking the SHELL environment variable.
        if os.environ.get("SHELL"):
            # Use Git Bash (or similar) to open a new terminal.
            bash_path = r"C:\Program Files\Git\bin\bash.exe"  # Adjust this path if needed.
            arg_str = " ".join(extra_args)
            cmd = f'start "" "{bash_path}" --login -i -c "python {simulation_script} {arg_str}"'
            subprocess.Popen(cmd, shell=True, env=env)
        else:
            # Default to Command Prompt.
            arg_str = " ".join(extra_args)
            cmd = f'start cmd /k "python {simulation_script} {arg_str}"'
            subprocess.Popen(cmd, shell=True, env=env)
    else:
        typer.echo("Unsupported OS for launching a new terminal or screen session.")
        raise typer.Exit(code=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator_id", required=True)
    parser.add_argument("--ports", required=True)  # comma-separated string of ports
    parser.add_argument("--env_type", required=True)
    parser.add_argument("--connection", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--total_agents", type=int, required=True)
    parser.add_argument("--url", default="")
    args = parser.parse_args()
    
    # Parse ports.
    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    from agent_gpt.utils.config_utils import load_config, save_config        
    from agent_gpt.env_host.server import EnvServer    
    # Load the latest configuration.
    config_data = load_config()
    
    # Create launchers for each port.
    launchers = []
    for port in ports:
        if args.connection == "tunnel":
            from agent_gpt.utils.tunnel import create_tunnel
            url = create_tunnel(port)
        else:
            url = args.url

        launcher = EnvServer.launch(args.env_type, url, args.host, port)
        launchers.append(launcher)
    
    # Distribute agents across launchers.
    num_launchers = len(launchers)
    base_agents = args.total_agents // num_launchers
    remainder = args.total_agents % num_launchers
    agents_array = [base_agents] * num_launchers
    for i in range(remainder):
        agents_array[i] += 1
    
    added_env_hosts = []
    # Update the hyperparameters config with new environment host entries.
    env_host = config_data.get("hyperparams", {}).get("env_host", {})
    for i, launcher in enumerate(launchers):
        key = f"{args.simulator_id}:{launcher.port}"
        env_endpoint = launcher.url if args.connection == "tunnel" else launcher.endpoint
        env_host[key] = {"env_endpoint": env_endpoint, "num_agents": agents_array[i]}
        added_env_hosts.append(key)
        typer.echo(f"Configured env_host entry: {env_endpoint} with {agents_array[i]} agents")
    
    config_data.setdefault("hyperparams", {})["env_host"] = env_host
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

    # After simulation ends, remove only the env_host entries added for this simulation.
    for key in added_env_hosts:
        env_host.pop(key, None)

    config_data["hyperparams"]["env_host"] = env_host
    save_config(config_data)

    if args.connection == "tunnel":
        from pyngrok import ngrok
        for launcher in launchers:
            try:
                ngrok.disconnect(launcher.url)
            except Exception:
                pass
    
    typer.echo("Simulation terminated.")

if __name__ == "__main__":
    main()
