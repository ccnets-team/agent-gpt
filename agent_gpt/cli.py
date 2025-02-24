import os
import sys
import yaml
import typer
from agent_gpt import AgentGPT
from config.sagemaker import SageMakerConfig
from config.hyperparams import Hyperparameters
from config.network import NetworkConfig  # <-- New import for network config
from typing import List

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

@app.command("config", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def config(ctx: typer.Context):
    """
    Update configuration settings.

    You can update any configuration parameter dynamically. For example:
        agent-gpt config --batch_size 64 --lr_init 0.0005 --env_id "CartPole-v1"
        agent-gpt config --network.host 1.2.3.4 --network.port 8080

    Use dot notation for nested fields (e.g., --exploration.continuous.initial_sigma 0.2).

    After updating the configuration, the command will also display any running simulation launcher ports.
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
    default_sagemaker = SageMakerConfig().to_dict()  # Now nested with "trainer" and "inference"
    default_network = NetworkConfig.from_network_info().to_dict()
    
    # Merge defaults with any stored overrides.
    full_config = {
        "network": deep_merge(default_network, stored_overrides.get("network", {})),
        "sagemaker": deep_merge(default_sagemaker, stored_overrides.get("sagemaker", {})),
        "hyperparams": deep_merge(default_hyperparams, stored_overrides.get("hyperparams", {})),
    }

    # Merge new changes into the full config.
    sections = ["network", "sagemaker", "hyperparams"]

    for key, value in new_changes.items():
        updated = False
        for section in sections:
            section_data = full_config.get(section, {})
            if key in section_data:
                # If the key exists in this section, merge/update it.
                if isinstance(value, dict) and isinstance(section_data.get(key), dict):
                    full_config[section][key] = deep_merge(section_data.get(key), value)
                else:
                    full_config[section][key] = value
                updated = True
                break
        # if not updated:
        #     # If the key isn't found in any section, add it to hyperparams by default.
        #     full_config.setdefault("hyperparams", {})[key] = value

    # Save the merged configuration back to disk.
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
    from env_host.local import LocalEnv  # Adjust the import if necessary

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
    sagemaker_trainer_conf = config_data.get("sagemaker", {})
    # Fallback: if role_arn is not provided in sagemaker-trainer, try hyperparams.
    if not sagemaker_trainer_conf.get("role_arn"):
        sagemaker_trainer_conf["role_arn"] = config_data.get("hyperparams", {}).get("role_arn")
    
    # Construct the SageMakerConfig for training with nested TrainerConfig.
    from config.sagemaker import SageMakerConfig, TrainerConfig
    sagemaker_config = SageMakerConfig(
        role_arn=sagemaker_trainer_conf.get("role_arn"),
        trainer=TrainerConfig(
            image_uri=sagemaker_trainer_conf.get("image_uri", TrainerConfig().image_uri),
            output_path=sagemaker_trainer_conf.get("output_path", TrainerConfig().output_path),
            instance_type=sagemaker_trainer_conf.get("instance_type", TrainerConfig().instance_type),
            instance_count=sagemaker_trainer_conf.get("instance_count", TrainerConfig().instance_count),
            max_run=sagemaker_trainer_conf.get("max_run", TrainerConfig().max_run)
        ),
        region=sagemaker_trainer_conf.get("region", "ap-northeast-2")
    )

    # Build training hyperparameters from the "hyperparams" section.
    hyperparams_conf = config_data.get("hyperparams", {})
    default_hyperparams = Hyperparameters().to_dict()
    full_hyperparams = deep_merge(default_hyperparams, hyperparams_conf)
    hyperparams_config = Hyperparameters(**full_hyperparams)

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

    # Use the sagemaker-inference configuration.
    sagemaker_inference_conf = config_data.get("sagemaker-inference", {})
    if not sagemaker_inference_conf.get("role_arn"):
        sagemaker_inference_conf["role_arn"] = config_data.get("hyperparams", {}).get("role_arn")
    
    from config.sagemaker import SageMakerConfig, InferenceConfig
    sagemaker_config = SageMakerConfig(
        role_arn=sagemaker_inference_conf.get("role_arn"),
        inference=InferenceConfig(
            image_uri=sagemaker_inference_conf.get("image_uri", InferenceConfig().image_uri),
            model_data=sagemaker_inference_conf.get("model_data", InferenceConfig().model_data),
            endpoint_name=sagemaker_inference_conf.get("endpoint_name", InferenceConfig().endpoint_name),
            instance_type=sagemaker_inference_conf.get("instance_type", InferenceConfig().instance_type),
            instance_count=sagemaker_inference_conf.get("instance_count", InferenceConfig().instance_count)
        ),
        region=sagemaker_inference_conf.get("region", "ap-northeast-2")
    )

    typer.echo("Deploying inference endpoint...")
    gpt_api = AgentGPT.infer(sagemaker_config)
    typer.echo(f"Inference endpoint deployed: {gpt_api.endpoint_name}")

if __name__ == "__main__":
    app()
