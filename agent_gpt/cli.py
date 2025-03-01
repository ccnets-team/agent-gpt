import logging
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

import os
import yaml
import typer
import requests
import json
from typing import List
from .core import AgentGPT
from .config.environment import MultiEnvironmentConfig 
from .config.network import NetworkConfig
from .config.hyperparams import Hyperparameters
from .config.sagemaker import SageMakerConfig
from .env_host.server import EnvServer
from typing import Optional

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

def recursive_update(target, changes: dict, prefix="") -> tuple:
    """
    Recursively update attributes of an object (or dictionary) using a nested changes dict.
    Only updates existing attributes/keys.

    Returns:
        tuple: (changed, diffs)
            changed (bool): True if any update was made, False otherwise.
            diffs (list): A list of differences in the form (full_key, old_value, new_value)
    """
    changed = False
    diffs = []

    if isinstance(target, dict):
        for k, v in changes.items():
            if k in target:
                current_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    sub_changed, sub_diffs = recursive_update(target[k], v, prefix=current_key)
                    if sub_changed:
                        changed = True
                        diffs.extend(sub_diffs)
                else:
                    if target[k] != v:
                        old_val = target[k]
                        target[k] = v
                        changed = True
                        diffs.append((current_key, old_val, v))
            # Do not add new keys.
    else:
        for attr, new_val in changes.items():
            if not hasattr(target, attr):
                continue
            current_val = getattr(target, attr)
            current_key = f"{prefix}.{attr}" if prefix else attr
            # If the new value is a dict, try to update the inner attributes.
            if isinstance(new_val, dict):
                sub_changed, sub_diffs = recursive_update(current_val, new_val, prefix=current_key)
                if sub_changed:
                    changed = True
                    diffs.extend(sub_diffs)
            else:
                if current_val != new_val:
                    old_val = current_val
                    setattr(target, attr, new_val)
                    changed = True
                    diffs.append((current_key, old_val, new_val))

    return changed, diffs

@app.command("register")
def register(
    account_id: str = typer.Argument(..., help="AWS account ID of the beta tester"),
    email: Optional[str] = typer.Option(None, help="Optional email address of the beta tester")
):
    """
    Register a new user account.
    
    This command sends a POST request to the API endpoint to register the beta tester.
    The API updates the trust policy of the AgentGPT-BetaTester role.
    """
    typer.echo(f"Registering beta tester with account ID: {account_id} ...")
    
    # Set the API endpoint URL (using non-custom API for now)
    url = "https://agentgpt-beta.ccnets.org"
    # Build the payload data with the required clientAccountId and optional email
    payload = {
        "clientAccountId": account_id,
    }
    if email:
        payload["Email"] = email
    
    # Set the headers for the POST request
    headers = {'Content-Type': 'application/json'}
    
    try:
        # Send the POST request to the API Gateway endpoint
        response = requests.post(url, json=payload, headers=headers)
        
        # Print the response status and body
        typer.echo(f"Status Code: {response.status_code}")
        typer.echo(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            # Parse the response JSON
            response_data = response.json()
            if response_data.get("statusCode") == 200:
                typer.echo("Registration successful.")
                stored_overrides = load_config()
                default_sagemaker = SageMakerConfig()
                default_sagemaker.set_config(**stored_overrides.get("sagemaker", {}))
                default_sagemaker.set_account_id(account_id)
                stored_overrides["sagemaker"] = default_sagemaker.to_dict()
                save_config(stored_overrides)
            else:
                typer.echo("Registration failed.")
        else:
            typer.echo("Error registering beta tester connection.")
    except Exception as e:  
        typer.echo(f"Error sending request: {e}")

@app.command("config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def config(ctx: typer.Context):
    """
    Update configuration settings.
    
    Modes:
    
      Field Update Mode:
        Example: agent-gpt config --batch_size 64 --lr_init 0.0005 --env_id "CartPole-v1"
    
      Method Mode:
        Example: agent-gpt config --set_env_host local0 http://your_domain.com:port 32
    
    Note:
      You can use dot notation to access nested configuration values.
      For example:
        agent-gpt config --sagemaker.trainer.max_run 360
      will update the 'max_run' value inside the 'trainer' configuration under 'sagemaker'.
      The top-level prefixes 'hyperparams', 'sagemaker', and 'network' can be omitted for convenience.
    
    Available Methods:
      set_region         - Set the AWS region for SageMaker configurations.
                           This method updates the ECR image URIs for both the trainer and inference images.
                           Only two regions are allowed: 'us-east-1' and 'ap-northeast-2' (Seoul).
      
      set_env_host       - Set a new environment host.
      del_env_host       - Delete an existing environment host.
      set_exploration    - Set exploration parameters.
      del_exploration    - Delete an exploration configuration.
      
      compose_environment  - Automatically generate a complete environment setup.
                           This command creates a Dockerfile with stored configs to your environment
                           by bundling the necessary files, installing dependencies, and
                           setting the appropriate entry point. It also generates a
                           corresponding Kubernetes manifest for deployment to AWS EKS.
    """

    # Parse CLI extra arguments into a nested dictionary.
    new_changes = parse_extra_args(ctx.args)
    
    # If no extra CLI arguments are provided, display the help for this command.
    if not ctx.args:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    # Load stored configuration overrides.
    stored_overrides = load_config()

    # Instantiate default configuration objects.
    default_environment = MultiEnvironmentConfig()
    default_network = NetworkConfig().from_network_info()
    default_hyperparams = Hyperparameters()
    default_sagemaker = SageMakerConfig()

    # Apply stored overrides.
    default_environment.set_config(**stored_overrides.get("environment", {}))
    default_network.set_config(**stored_overrides.get("network", {}))
    default_hyperparams.set_config(**stored_overrides.get("hyperparams", {}))
    default_sagemaker.set_config(**stored_overrides.get("sagemaker", {}))

    # Use the updated defaults as our working configuration.
    new_environment = default_environment
    new_network = default_network
    new_hyperparams = default_hyperparams 
    new_sagemaker = default_sagemaker 

    # List to collect change summaries.
    list_changes = []

    # Loop through the parsed changes.
    for key, value in new_changes.items():
        diffs_for_key = []
        changed = False
        # If the top-level key is one of the config namespaces and the value is a dict with one item,
        # then unpack that item so that:
        #   key becomes the inner attribute (e.g. "exploration")
        #   value becomes its nested dict (e.g. {'continuous': {'mu': 32}})
        if key in ("environment", "hyperparams", "sagemaker", "network") and isinstance(value, dict) and len(value) == 1:
            inner_key, inner_value = list(value.items())[0]
            key = inner_key
            value = inner_value

        # Otherwise, update all config objects that have the attribute.
        for obj in [new_environment, new_network, new_hyperparams, new_sagemaker]:
            if not hasattr(obj, key):
                continue
            attr = getattr(obj, key)
            if callable(attr):
                if not isinstance(value, list):
                    value = [value]
                # Filter out None values if necessary.
                converted_args = [parse_value(arg) for arg in value if arg is not None]
                if converted_args:
                    attr(*converted_args)
                else:
                    attr()
                arg_str = " ".join(str(x) for x in converted_args)
                diffs_for_key.append((key, None, arg_str))
                changed = True
            elif isinstance(value, dict):
                ch, diffs = recursive_update(attr, value, prefix=key)
                if ch:
                    changed = True
                    diffs_for_key.extend(diffs)
            else:
                current_val = getattr(obj, key)
                if current_val != value:
                    setattr(obj, key, value)
                    changed = True
                    diffs_for_key.append((key, current_val, value))
        list_changes.append((key, value, changed, diffs_for_key))

    # Print detailed change summaries.
    for key, value, changed, diffs in list_changes:
        if changed:
            for full_key, old_val, new_val in diffs:
                if old_val is None:
                    typer.echo(typer.style(
                        f" - {full_key} {new_val}",
                        fg=typer.colors.GREEN
                    ))
                else:
                    typer.echo(typer.style(
                        f" - {full_key} changed from {old_val} to {new_val}",
                        fg=typer.colors.GREEN
                    ))
        else:
            typer.echo(typer.style(
                f" - {key}: no changes applied (already up-to-date or attribute/method not found)",
                fg=typer.colors.YELLOW
            ))

    full_config = {
        "environment": default_environment.to_dict(),
        "network": default_network.to_dict(),
        "hyperparams": default_hyperparams.to_dict(),
        "sagemaker": default_sagemaker.to_dict(),
    }
    save_config(full_config)

def get_default(section: str) -> dict:
    if section == "environment":
        return MultiEnvironmentConfig().to_dict()
    elif section == "network":
        return NetworkConfig().from_network_info().to_dict()
    elif section == "hyperparams":
        return Hyperparameters().to_dict()
    elif section == "sagemaker":
        return SageMakerConfig().to_dict()
    else:
        return {}

@app.command("clear")
def clear_config(
    section: Optional[str] = typer.Argument(
        None,
        help="Optional configuration section to clear (environment, network, hyperparams, sagemaker). If not provided, clears the entire configuration."
    )
):
    """
    Clear configuration settings. If a section is provided, reset that section to its default.
    Otherwise, delete the entire configuration file from disk.
    """
    allowed_sections = {"environment", "network", "hyperparams", "sagemaker"}
    if section:
        if section not in allowed_sections:
            typer.echo(f"Invalid section '{section}'. Allowed sections: {', '.join(allowed_sections)}.")
            raise typer.Exit()
        config_data = load_config()
        config_data[section] = get_default(section)
        save_config(config_data)
        typer.echo(f"Configuration section '{section}' has been reset to default.")
    else:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            os.remove(DEFAULT_CONFIG_PATH)
            typer.echo("Entire configuration file deleted from disk.")
        else:
            typer.echo("No configuration file found to delete.")
    
@app.command("list")
def list_config(
    section: Optional[str] = typer.Argument(
        None,
        help="Configuration section to list (environment, network, hyperparams, sagemaker). If not provided, lists all configuration settings."
    )
):
    """
    List the current configuration settings. If a section is provided,
    only that part of the configuration is displayed.
    """
    config_data = load_config()
    
    # If no configuration exists, generate defaults and save them.
    if not config_data:
        default_environment = MultiEnvironmentConfig().to_dict()
        default_network = NetworkConfig().from_network_info().to_dict()
        default_hyperparams = Hyperparameters().to_dict()
        default_sagemaker = SageMakerConfig().to_dict()
        config_data = {
            "environment": default_environment,
            "network": default_network,
            "hyperparams": default_hyperparams,
            "sagemaker": default_sagemaker,
        }
        save_config(config_data)
    
    if section:
        # Retrieve the specified section and print its contents directly.
        section_data = config_data.get(section, {})
        typer.echo(f"Current configuration for '{section}':")
        typer.echo(yaml.dump(section_data, default_flow_style=False))
    else:
        typer.echo("Current configuration:")
        # Iterate over each top-level section and print them without extra nesting.
        for sec, sec_data in config_data.items():
            typer.echo(f"**{sec}**:")
            typer.echo(yaml.dump(sec_data, default_flow_style=False))
    
@app.command("simulate")
def simulate(
    env_identifier: str = typer.Argument(
        "local",
        help="Environment identifier to simulate. Default: 'local'."
    ),
    ports: List[int] = typer.Argument(
        ..., 
        help="One or more container port numbers on which to run the simulation server. Example: 80"
    )
):
    """
    Launch an environment simulation locally using the specified simulator and ports.

    Examples:
      agent-gpt simulate gym 80
      agent-gpt simulate local 80
      agent-gpt simulate unity 8080

    This command starts a simulation server for the specified environment on each provided port.
    Press Ctrl+C (or CTRL+C on Windows) to terminate the simulation.
    """

    # Load configuration to get the network settings.
    config_data = load_config()
    network_conf = config_data.get("network", {})
    host = network_conf.get("host", "localhost")
    ip = network_conf.get("public_ip", network_conf.get("internal_ip", "127.0.0.1"))

    multi_environment_conf = config_data.get("environment", {}).get("envs", {})
    environment_conf = multi_environment_conf.get(env_identifier, {})
    env = environment_conf.get("env")
    env_id = environment_conf.get("env_id")
    entry_point = environment_conf.get("entry_point")
    host_type = environment_conf.get("host_type")
    
    if env == "unity" and not entry_point:
        typer.echo("Unity environment requires an entry point to launch the simulation.")
        raise typer.Exit(code=1)
    
    if host_type == "local":
        launchers = []
        # Get the port mappings; if a port is not mapped, use the provided port directly.
        for port in ports:
            launcher = EnvServer.launch(
                env=env,
                env_id=env_id,
                entry_point=entry_point,
                ip=ip,
                host=host,
                port=port
            )
            launchers.append(launcher)

        import sys
        if sys.platform.startswith("win"):
            typer.echo("Simulation running. Press CTRL+C to terminate the simulation...")
        else:
            typer.echo("Simulation running. Press Ctrl+C to terminate the simulation...")

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

        typer.echo("Local environments launched on:")
        for launcher in launchers:
            typer.echo(f" - {launcher.public_ip}:{launcher.port}")
    else:
        typer.echo("Other host types are not supported yet.")
        
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
