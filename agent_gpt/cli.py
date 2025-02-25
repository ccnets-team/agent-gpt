import os
import yaml
import typer
from typing import List
from .core import AgentGPT
from .config.sagemaker import SageMakerConfig
from .config.hyperparams import Hyperparameters
from .config.network import NetworkConfig
from .env_host.local import LocalEnv

app = typer.Typer()

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.agent_gpt/config.yaml")

def load_config() -> dict:
    """Load the saved configuration overrides."""
    if os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_config(config_data: dict) -> None:
    """Save configuration overrides to disk."""
    os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
    with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

def parse_value(value: str):
    """
    Try converting the string to int, float, or bool.
    If all conversions fail, return the string.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    if value is not None:
        lower = value.lower()
        if lower in ["true", "false"]:
            return lower == "true"
    return value

def deep_merge(default: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries.
    Values in 'override' update those in 'default'.
    """
    merged = default.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def parse_extra_args(args: list[str]) -> dict:
    """
    Parses extra CLI arguments provided in the form:
      --key value [value ...]
    Supports nested keys via dot notation, e.g.:
      --env_hosts.local1.env_endpoint "http://example.com:8500"
    Returns a nested dictionary of the parsed values.
    """
    new_changes = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # remove the leading "--"
            i += 1
            # Gather all subsequent arguments that do not start with '--'
            values = []
            while i < len(args) and not args[i].startswith("--"):
                values.append(args[i])
                i += 1

            # Determine if we have no values, a single value, or multiple values.
            if not values:
                parsed_value = None
            elif len(values) == 1:
                parsed_value = parse_value(values[0])
            else:
                parsed_value = [parse_value(val) for val in values]

            # Build a nested dictionary using dot notation.
            keys = key.split(".")
            d = new_changes
            for sub_key in keys[:-1]:
                d = d.setdefault(sub_key, {})
            d[keys[-1]] = parsed_value
        else:
            i += 1
    return new_changes

def recursive_update(target, changes: dict) -> bool:
    """
    Recursively update attributes of an object (or dictionary) using a nested changes dict.
    Only updates existing attributes/keys.
    
    Returns:
        bool: True if any update was made, False otherwise.
    """
    changed = False

    if isinstance(target, dict):
        for k, v in changes.items():
            if k in target:
                if isinstance(target[k], dict) and isinstance(v, dict):
                    if recursive_update(target[k], v):
                        changed = True
                else:
                    if target[k] != v:
                        target[k] = v
                        changed = True
            # Do not add new keys.
    else:
        for attr, new_val in changes.items():
            if not hasattr(target, attr):
                continue
            current_val = getattr(target, attr)
            if isinstance(current_val, dict) and isinstance(new_val, dict):
                if recursive_update(current_val, new_val):
                    changed = True
            else:
                if current_val != new_val:
                    setattr(target, attr, new_val)
                    changed = True

    return changed


@app.command("config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def config(ctx: typer.Context):
    """
    Update configuration settings.

    Modes:

      Field Update Mode:
        Example: agent-gpt config --batch_size 64 --lr_init 0.0005 --env_id "CartPole-v1"

      Method Mode:
        Example: agent-gpt config --set_env_host local0 http://your_domain.com 32

    Note:
      You can use dot notation to access nested configuration values.
      For example:
        agent-gpt config --sagemaker.trainer.max_run 360
      will update the 'max_run' value inside the 'trainer' configuration under 'sagemaker'.
      The top-level prefixes 'hyperparams', 'sagemaker', and 'network'
      can be omitted for convenience.

    Available Methods:

      set_env_host       - Set a new environment host.
      del_env_host       - Delete an existing environment host.
      set_exploration    - Set exploration parameters.
      del_exploration    - Delete an exploration configuration.
    """

    # Parse CLI extra arguments into a nested dictionary.
    new_changes = parse_extra_args(ctx.args)
    
    if not new_changes:
        typer.echo("No configuration options provided.")
        raise typer.Exit()

    # Load stored configuration overrides.
    stored_overrides = load_config()
    
    # Instantiate default configuration objects.
    default_hyperparams = Hyperparameters()
    default_sagemaker = SageMakerConfig()
    default_network = NetworkConfig().from_network_info()

    # Apply stored overrides.
    default_hyperparams.set_config(**stored_overrides.get("hyperparams", {}))
    default_sagemaker.set_config(**stored_overrides.get("sagemaker", {}))
    default_network.set_config(**stored_overrides.get("network", {}))
    
    # List to collect change summaries: (key, value, changed_flag)
    list_changes = []
    
    # Loop through the parsed changes.
    for key, value in new_changes.items():
        changed = False
        # Top-level namespaces
        if key == "hyperparams":
            changed = recursive_update(default_hyperparams, value)
        elif key == "sagemaker":
            changed = recursive_update(default_sagemaker, value)
        elif key == "network":
            changed = recursive_update(default_network, value)
        else:
            # Otherwise, update all config objects that have the attribute.
            for obj in [default_hyperparams, default_sagemaker, default_network]:
                if not hasattr(obj, key):
                    continue
                attr = getattr(obj, key)
                updated = False
                if callable(attr):
                    # Call the method if applicable.
                    if not isinstance(value, list):
                        value = [value]
                    converted_args = [parse_value(arg) for arg in value]
                    attr(*converted_args)
                    updated = True
                elif isinstance(value, dict):
                    updated = recursive_update(attr, value)
                else:
                    if getattr(obj, key) != value:
                        setattr(obj, key, value)
                        updated = True
                changed = changed or updated

        list_changes.append((key, value, changed))

    # Print only a summary of changes instead of the full configuration.
    for key, value, changed in list_changes:
        if changed:
            typer.echo(typer.style(f" - {key} updated to {value}", fg=typer.colors.GREEN))
        else:
            typer.echo(typer.style(f" - {key} not updated (no matching attribute)", fg=typer.colors.YELLOW))

    full_config = {
        "hyperparams": default_hyperparams.to_dict(),
        "sagemaker": default_sagemaker.to_dict(),
        "network": default_network.to_dict(),
    }
    save_config(full_config)

@app.command("clear")
def clear_config():
    """
    Delete the configuration cache.
    """
    if os.path.exists(DEFAULT_CONFIG_PATH):
        os.remove(DEFAULT_CONFIG_PATH)
        typer.echo("Configuration cache deleted.")
    else:
        typer.echo("No configuration cache found.")

@app.command("list")
def list_config():
    """
    List the current configuration settings.
    """
    config_data = load_config()
    typer.echo("Current configuration:")
    typer.echo(yaml.dump(config_data))
    
@app.command("simulate")
def simulate(
    env: str = typer.Argument(
        ..., 
        help="Name of the environment simulator to use. For example: 'gym' or 'unity'."
    ),
    ports: List[int] = typer.Argument(
        ..., 
        help="One or more port numbers on which to run the local simulation server. Example: 8000 8001"
    )
):
    """
    Launch an environment simulation locally using the specified simulator and ports.

    Examples:
      agent-gpt simulate gym 8000 8001
      agent-gpt simulate unity 8500

    This command starts a simulation server for the specified environment on each port.
    Press Ctrl+C (or CTRL+C on Windows) to terminate the simulation.
    """

    # Load configuration to get the network settings.
    config_data = load_config()
    network_conf = config_data.get("network", {})
    host = network_conf.get("host", "localhost")  # Default to 'localhost'
    ip = network_conf.get("public_ip", network_conf.get("internal_ip", "127.0.0.1"))
    
    launchers = []
    # Launch the simulation server on each specified port.
    for port in ports:
        launcher = LocalEnv.launch(
            env=env,
            ip=ip,
            host=host,
            port=port
        )
        launchers.append(launcher)

    # Echo termination instructions based on OS.
    import sys
    if sys.platform.startswith("win"):
        typer.echo("Simulation running. Press CTRL+C to terminate the simulation...")
    else:
        typer.echo("Simulation running. Press Ctrl+C to terminate the simulation...")

    # Block the main thread by joining server threads in a loop with a timeout.
    try:
        while any(launcher.server_thread.is_alive() for launcher in launchers):
            for launcher in launchers:
                # Join with a short timeout so we can periodically check for KeyboardInterrupt.
                launcher.server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        typer.echo("Shutdown requested, stopping all servers...")
        for launcher in launchers:
            launcher.shutdown()
        # Optionally wait a little longer for threads to shutdown gracefully.
        for launcher in launchers:
            launcher.server_thread.join(timeout=2)

    typer.echo("Local environments launched on:")
    for launcher in launchers:
        typer.echo(f" - {launcher.public_ip}")

@app.command()
def train():
    """
    Launch a SageMaker training job for AgentGPT using configuration settings.
    This command loads training configuration from the saved config file.
    """
    config_data = load_config()

    # Use the sagemaker-trainer configuration.
    sagemaker_conf = config_data.get("sagemaker", {})
    hyperparams_conf = config_data.get("hyperparams", {})

    sagemaker_config = SageMakerConfig(**sagemaker_conf)
    hyperparams_config = Hyperparameters(**hyperparams_conf)

    typer.echo("Submitting training job...")
    estimator = AgentGPT.train(sagemaker_config, hyperparams_config)
    typer.echo(f"Training job submitted: {estimator.latest_training_job.name}")

@app.command()
def infer():
    """
    Deploy or reuse a SageMaker inference endpoint for AgentGPT using configuration settings.
    This command loads inference configuration from the saved config file.
    """
    config_data = load_config()

    sagemaker_conf = config_data.get("sagemaker", {})
    sagemaker_config = SageMakerConfig(**sagemaker_conf)

    typer.echo("Deploying inference endpoint...")
    
    gpt_api = AgentGPT.infer(sagemaker_config)
    typer.echo(f"Inference endpoint deployed: {gpt_api.endpoint_name}")

if __name__ == "__main__":
    app()   
