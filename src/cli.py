import os
import yaml
import typer
from agent_gpt import AgentGPT
from config.sagemaker_config import SageMakerConfig
from config.hyperparams import Hyperparameters

app = typer.Typer()

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.agent_gpt/config.yaml")
LAUNCHERS_FILE = os.path.expanduser("~/.agent_gpt/launchers.yaml")

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


@app.command("config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def config(ctx: typer.Context):
    """
    Update configuration settings.

    You can update any configuration parameter dynamically. For example:

        agent-gpt config --batch_size 64 --lr_init 0.0005 --env_id "CartPole-v1"

    Use dot notation for nested fields (e.g., --exploration.continuous.initial_sigma 0.2).
    """
    # Parse extra arguments manually (expecting --key value pairs)
    args = ctx.args
    new_changes = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # remove the leading "--"
            if i + 1 < len(args):
                value = args[i + 1]
                i += 2
            else:
                value = None
                i += 1
            parsed_value = parse_value(value)
            # Support nested keys via dot notation.
            keys = key.split(".")
            d = new_changes
            for sub_key in keys[:-1]:
                d = d.setdefault(sub_key, {})
            d[keys[-1]] = parsed_value
        else:
            i += 1

    # Load any stored configuration (overrides) from file.
    stored_overrides = load_config()

    # Load the full defaults from your dataclasses.
    default_hyperparams = Hyperparameters().to_dict()
    default_sagemaker = SageMakerConfig().to_dict()

    # Merge defaults with any stored overrides.
    full_config = {
        "hyperparams": deep_merge(default_hyperparams, stored_overrides.get("hyperparams", {})),
        "sagemaker": deep_merge(default_sagemaker, stored_overrides.get("sagemaker", {}))
    }

    # Merge in the new changes.
    full_config["hyperparams"] = deep_merge(full_config["hyperparams"], new_changes)

    # Save the full, merged configuration back to disk.
    save_config(full_config)

    typer.echo("Updated configuration:")
    typer.echo(yaml.dump(full_config))


@app.command("list")
def list_config():
    """
    List the full effective configuration, merging defaults with stored overrides.
    If no overrides exist, the default configuration is printed.
    """
    # Load stored configuration overrides.
    config_overrides = load_config()

    # Load defaults from your dataclasses.
    default_hyperparams = Hyperparameters().to_dict()
    default_sagemaker = SageMakerConfig().to_dict()

    # Merge defaults with overrides.
    merged_hyperparams = deep_merge(default_hyperparams, config_overrides.get("hyperparams", {}))
    merged_sagemaker = deep_merge(default_sagemaker, config_overrides.get("sagemaker", {}))
    full_config = {"hyperparams": merged_hyperparams, "sagemaker": merged_sagemaker}

    typer.echo("Full configuration:")
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

# --- Launcher Persistence Helpers ---
def load_launchers() -> list:
    """Load a list of persisted launcher PIDs."""
    if os.path.exists(LAUNCHERS_FILE):
        with open(LAUNCHERS_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []
    return []

def save_launchers(launchers: list) -> None:
    """Save the list of launcher PIDs to disk."""
    os.makedirs(os.path.dirname(LAUNCHERS_FILE), exist_ok=True)
    with open(LAUNCHERS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(launchers, f)
# --- Other commands remain unchanged ---

@app.command("simulate")
def simulate(
    ip: str = typer.Option(None, help="Local IP address for simulation (use with --port)"),
    port: int = typer.Option(None, help="Port for simulation (use with --ip)"),
    tunnel: str = typer.Option(None, help="Tunnel URL for simulation (ngrok, etc.)"),
    endpoint: str = typer.Option(None, help="Endpoint URL for simulation (e.g., for eks)")
):
    """
    Update the configuration with a new environment host and launch simulation if applicable.

    Exactly one of the following modes must be provided:
      - Local mode: provide both --ip and --port.
      - Tunnel mode: provide --tunnel.
      - Endpoint mode: provide --endpoint.

    In local mode, the simulation is launched and its launcher PID is stored persistently.
    """
    mode_count = 0
    if ip is not None or port is not None:
        mode_count += 1
    if tunnel is not None:
        mode_count += 1
    if endpoint is not None:
        mode_count += 1
    if mode_count != 1:
        typer.echo("Error: Provide exactly one mode: (--ip with --port) OR (--tunnel) OR (--endpoint).")
        raise typer.Exit(code=1)

    if ip is not None or port is not None:
        if ip is None or port is None:
            typer.echo("Error: Both --ip and --port must be provided for local simulation.")
            raise typer.Exit(code=1)
        env_endpoint_value = f"http://{ip}:{port}"
        mode = "local-ip"
    elif tunnel is not None:
        env_endpoint_value = tunnel
        mode = "tunnel"
    elif endpoint is not None:
        env_endpoint_value = endpoint
        mode = "endpoint"

    # --- Update Configuration ---
    config_data = load_config()
    default_hyperparams = Hyperparameters().to_dict()
    merged_hyperparams = deep_merge(default_hyperparams, config_data.get("hyperparams", {}))

    from config.hyperparams import EnvHost
    from dataclasses import asdict
    new_env_host = EnvHost(env_endpoint=env_endpoint_value, num_agents=128)
    merged_hyperparams.setdefault("env_hosts", {})[mode] = asdict(new_env_host)
    config_data["hyperparams"] = merged_hyperparams
    save_config(config_data)
    typer.echo(f"Environment host '{mode}' with endpoint '{env_endpoint_value}' added to configuration.")

    # --- Launch Simulation if in local-ip mode ---
    if mode == "local-ip":
        from src.env_host.simulator import EnvSimulator
        launcher = EnvSimulator.launch_on_local_with_ip(
            env_simulator="gym",
            ip_address=ip,
            host=ip,
            port=port
        )
        # Persist the launcher PID (or other identifier) to a file so that it can be terminated later.
        launchers = load_launchers()
        # Here, we assume that 'launcher' has a 'pid' attribute.
        launchers.append(launcher.pid)
        save_launchers(launchers)
        typer.echo(f"Local environment launched on http://{ip}:{port} (PID: {launcher.pid}).")
    else:
        typer.echo("Simulation launch is not implemented for tunnel/endpoint modes.")

@app.command()
def train():
    """
    Launch a SageMaker training job for AgentGPT using configuration settings.
    
    This command no longer takes any command-line input.
    It loads training configuration from the saved config file.
    """
    config_data = load_config()
    
    # Build SageMaker configuration from the "sagemaker" section.
    sagemaker_conf = config_data.get("sagemaker", {})
    sagemaker_config = SageMakerConfig(
        role_arn=sagemaker_conf.get("role_arn"),
        image_uri=sagemaker_conf.get("image_uri"),
        model_data=sagemaker_conf.get("model_data"),
        output_path=sagemaker_conf.get("output_path"),
        instance_type=sagemaker_conf.get("instance_type"),
        instance_count=sagemaker_conf.get("instance_count"),
        region=sagemaker_conf.get("region"),
        max_run=sagemaker_conf.get("max_run")
    )
    
    # Build training hyperparameters from the "hyperparams" section.
    hyperparams_conf = config_data.get("hyperparams", {})
    default_hyperparams = Hyperparameters().to_dict()
    full_hyperparams = deep_merge(default_hyperparams, hyperparams_conf)
    # Create a Hyperparameters instance from the full configuration.
    hyperparams_config = Hyperparameters(**full_hyperparams)
    
    typer.echo("Submitting training job...")
    estimator = AgentGPT.train(sagemaker_config, hyperparams_config)
    typer.echo(f"Training job submitted: {estimator.latest_training_job.name}")

@app.command()
def infer():
    """
    Deploy or reuse a SageMaker inference endpoint for AgentGPT using configuration settings.
    
    This command no longer takes any command-line input.
    It loads inference configuration from the saved config file.
    """
    config_data = load_config()
    infer_conf = config_data.get("infer", {})
    sagemaker_config = SageMakerConfig(
        role_arn=infer_conf.get("role_arn"),
        image_uri=infer_conf.get("image_uri"),
        model_data=infer_conf.get("model_data"),
        endpoint_name=infer_conf.get("endpoint_name", "agent-gpt-inference-endpoint"),
        instance_type=infer_conf.get("instance_type"),
        instance_count=infer_conf.get("instance_count"),
        region=infer_conf.get("region")
    )
    typer.echo("Deploying inference endpoint...")
    gpt_api = AgentGPT.infer(sagemaker_config)
    typer.echo(f"Inference endpoint deployed: {gpt_api.endpoint_name}")

if __name__ == "__main__":
    app()
