import os
import yaml
import typer
from typing import List
from .core import AgentGPT
from .config.sagemaker import SageMakerConfig, TrainerConfig, InferenceConfig
from .config.hyperparams import Hyperparameters, EnvHost
from .config.network import NetworkConfig
from .env_host.local import LocalEnv  # Adjust the import if necessary

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

@app.command("config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def config(ctx: typer.Context):
    """
    Update configuration settings.

    **Field Update Mode:**
      Example: agent-gpt config --batch_size 64 --lr_init 0.0005 --env_id "CartPole-v1"

    **Method Mode:**
      Example: agent-gpt config --set_env_host local0 http://your_domain.com 32
    """
    # Parse CLI extra arguments into a dictionary.
    new_changes = parse_extra_args(ctx.args)
    
    # Load stored configuration overrides.
    stored_overrides = load_config()
    
    # Instantiate default configuration objects.
    default_hyperparams = Hyperparameters()
    default_sagemaker = SageMakerConfig()
    default_network = NetworkConfig().from_network_info()

    # Apply stored overrides to the defaults.
    default_hyperparams.set_config(**stored_overrides.get("hyperparams", {}))
    default_sagemaker.set_config(**stored_overrides.get("sagemaker", {}))
    default_network.set_config(**stored_overrides.get("network", {}))
    
    # Loop through the parsed changes.
    # For each key, if the config object has a callable with that name, call it;
    # otherwise, treat it as a normal field update.
    for key, value in new_changes.items():
        for obj in [default_hyperparams, default_sagemaker, default_network]:
            attr = getattr(obj, key, None)
            if callable(attr):
                # If value is not a list (e.g. a single value), wrap it.
                if not isinstance(value, list):
                    value = [value]
                # Optionally, you might want to convert values further here.
                converted_args = [parse_value(arg) for arg in value]
                attr(*converted_args)
            elif hasattr(obj, key):
                setattr(obj, key, value)
    
    # Build the full configuration dictionary.
    full_config = {
        "hyperparams": default_hyperparams.to_dict(),
        "sagemaker": default_sagemaker.to_dict(),
        "network": default_network.to_dict(),
    }
    
    # Save the merged configuration.
    save_config(full_config)
    
    typer.echo("Updated configuration:")
    typer.echo(yaml.dump(full_config))

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
