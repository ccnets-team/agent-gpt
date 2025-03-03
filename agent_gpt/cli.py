import warnings
import logging

# Suppress specific pydantic warning about the "json" field.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "json" in "MonitoringDatasetFormat" shadows an attribute in parent "Base"',
    category=UserWarning,
    module="pydantic._internal._fields"
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)

import typer
import os
import re
import yaml
import requests
from typing import Optional
from .config.simulator import SimulatorConfig 
from .config.hyperparams import Hyperparameters
from .config.sagemaker import SageMakerConfig
from .env_host.server import EnvServer
from .core import AgentGPT
from .utils.config_utils import load_config, save_config, generate_section_config, handle_config_method
from .utils.config_utils import convert_to_objects, parse_extra_args, initialize_config, apply_config_updates
from .utils.config_utils import DEFAULT_CONFIG_PATH, TOP_CONFIG_CLASS_MAP
import yaml

app = typer.Typer(add_completion=False, invoke_without_command=True)

def load_help_texts(yaml_filename: str) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, yaml_filename)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def auto_format_help(text: str) -> str:
    formatted = re.sub(r'([.:])\s+', r'\1\n\n', text)
    return formatted

help_texts = load_help_texts("help_config.yaml")

@app.command(
    "config",
    short_help=help_texts["config"]["short_help"],
    help=auto_format_help(help_texts["config"]["detailed_help"]),
)
def config(ctx: typer.Context):
    if not ctx.args:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    # Load stored configuration overrides.
    stored_overrides = load_config()
    config_obj = convert_to_objects(stored_overrides)
    diffs = []
    # Check if the first argument starts with "--" (field update mode) or not (method mode)
    if ctx.args[0].startswith("--"):
        new_changes = parse_extra_args(ctx.args)
        list_changes = apply_config_updates(config_obj, new_changes)
    else:
        list_changes = handle_config_method(ctx.args, config_obj)

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
            for full_key, old_val, new_val in diffs:
                typer.echo(typer.style(
                    f" - {key}: no changes applied {new_val}",
                    fg=typer.colors.YELLOW
                ))
            
    full_config = {}
    for key, obj in config_obj.items():
        full_config[key] = obj.to_dict()
    save_config(full_config)

@app.command(
    "edit",
    short_help=help_texts["edit"]["short_help"],
    help=auto_format_help(help_texts["edit"]["detailed_help"]),
)
def edit_config():
    # Check if the configuration file exists; if not, create a default one.
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        typer.echo("Configuration file not found. Creating a default configuration file...")
        default_config = initialize_config()
        save_config(default_config)
    
    try:
        import platform
        import subprocess
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["notepad.exe", DEFAULT_CONFIG_PATH])
        elif system == "Darwin":
            subprocess.Popen(["open", DEFAULT_CONFIG_PATH])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", DEFAULT_CONFIG_PATH])
        else:
            typer.launch(DEFAULT_CONFIG_PATH)
    except Exception as e:
        typer.echo(f"Failed to open the configuration file: {e}")

@app.command(
    "clear",
    short_help=help_texts["clear"]["short_help"],
    help=auto_format_help(help_texts["clear"]["detailed_help"]),
)
def clear_config(
    section: Optional[str] = typer.Argument(
        None,
    )
):
    allowed_sections = set(TOP_CONFIG_CLASS_MAP.keys())
    if section:
        if section not in allowed_sections:
            typer.echo(f"Invalid section '{section}'. Allowed sections: {', '.join(allowed_sections)}.")
            raise typer.Exit()
        config_data = load_config()
        config_data[section] = generate_section_config(section)
        save_config(config_data)
        typer.echo(f"Configuration section '{section}' has been reset to default.")
    else:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            os.remove(DEFAULT_CONFIG_PATH)
            typer.echo("Entire configuration file deleted from disk.")
        else:
            typer.echo("No configuration file found to delete.")

@app.command(
    "list",
    short_help=help_texts["list"]["short_help"],
    help=auto_format_help(help_texts["list"]["detailed_help"]),
)
def list_config(
    section: Optional[str] = typer.Argument(
        None,
    )
):
    config_data = load_config()
    
    # If no configuration exists, generate defaults and save them.
    if not config_data:
        config_data = initialize_config()
        save_config(config_data)
        
    if section:
        # Retrieve the specified section and print its contents directly.
        section_data = config_data.get(section, {})
        typer.echo(f"Current configuration for '{section}':")
        typer.echo(yaml.dump(section_data, default_flow_style=False, sort_keys=False))
    else:
        typer.echo("Current configuration:")
        for sec in TOP_CONFIG_CLASS_MAP.keys():
            if sec in config_data:
                typer.echo(f"**{sec}**:")
                typer.echo(yaml.dump(config_data[sec], default_flow_style=False, sort_keys=False))

@app.command(
    "upload",
    short_help=help_texts["upload"]["short_help"],
    help=auto_format_help(help_texts["upload"]["detailed_help"])
)
def upload(
    simulator_id: str = typer.Argument(
    )
):
    config_data = load_config()
    region = config_data.get("sagemaker", {}).get("region")
    if not region:
        typer.echo("Error: AWS region not set in the configuration.")
        raise typer.Exit(code=1)

    simulator_registry_data = config_data.get("simulator_registry", {})
    simulator = simulator_registry_data.get("simulator", {})
    simulator_data = simulator.get(simulator_id)
    if not simulator_data:
        typer.echo(f"Warning: No simulator config found for identifier '{simulator_id}'")
        raise typer.Exit(code=1)
    hosting = simulator_data.get("hosting")
    if hosting != "cloud":
        typer.echo(f"Error: Simulator '{simulator_id}' is not set up for cloud deployment.")
        raise typer.Exit(code=1)

    simulator_config = SimulatorConfig()
    simulator_config.set_config(**simulator_data)
    from .env_host.upload import upload_simulator
    
    try: 
        upload_simulator(region, simulator_config)
        typer.echo(f"Simulator '{simulator_id}' uploaded successfully.")
    except Exception as e:
        typer.echo(f"Error uploading simulator '{simulator_id}': {e}")
        raise typer.Exit(code=1)
    
    simulator_registry_data["simulator"][simulator_id] = simulator_config.to_dict()
    config_data["simulator_registry"] = simulator_registry_data
    save_config(config_data)

@app.command(
    "simulate",
    short_help=help_texts["simulate"]["short_help"],
    help=auto_format_help(help_texts["simulate"]["detailed_help"])
)
def simulate(
    simulator_id: str = typer.Argument("local", help="Simulator identifier (default: local)"),
    ports: list[int] = typer.Argument(None, help="List of port numbers")
):
    # Load configuration to get the network settings.
    config_data = load_config()

    simulator_registry_data = config_data.get("simulator_registry", {})
    simulator = simulator_registry_data.get("simulator", {})
    simulator_data = simulator.get(simulator_id)
    
    simulator_obj = SimulatorConfig()
    simulator_obj.set_config(**simulator_data)
        
    env_type = simulator_obj.env_type
    hosting = simulator_obj.hosting
    url = simulator_obj.url
    host = simulator_obj.host
    connection = simulator_obj.connection
    total_agents = simulator_obj.total_agents
    
    if not ports:
        typer.echo("No port numbers provided. Attempting to retrieve ports from the simulator configuration.")
        ports = simulator_obj.ports
    if not ports:
        typer.echo("Error: No available ports found. Please specify one or more port numbers.")
        raise typer.Exit(code=1)

    if hosting == "local":
            
        launchers = []
        for port in ports:
            if connection == "tunnel":
                from .utils.tunnel import create_tunnel
                url = create_tunnel(port)
            launcher = EnvServer.launch(
                env_type=env_type,
                url=url,
                host=host,
                port=port
            )
            launchers.append(launcher)
        
        num_launcers = len(launchers)
        base_agents = total_agents // num_launcers
        remainder = total_agents % num_launcers
        agents_array = [base_agents] * num_launcers
        for i in range(remainder):
            agents_array[i] += 1
        
        # Add environment hosts for the simulation.
        env_host = config_data.get("hyperparams", {}).get("env_host", {})
        added_env_hosts = []
        # Store simulation host info using a key like f"{simulator_id}:{port}"
        for i, launcher in enumerate(launchers):
            key = f"{simulator_id}:{launcher.port}"
            if connection == "tunnel":
                env_endpoint = launcher.url
            else:
                env_endpoint = launcher.endpoint
            env_host[key] = {"env_endpoint": env_endpoint, "num_agents": agents_array[i]}
            added_env_hosts.append(key)
            typer.echo(f"env_endpoint: {env_endpoint}, num_agents: {agents_array[i]}")

        # Update and save the config.
        config_data.setdefault("hyperparams", {})["env_host"] = env_host
        save_config(config_data)

        # Inform the user that the simulation command will block this terminal.
        typer.echo("Simulation running. This terminal is now dedicated to simulation; open another terminal for AgentGPT training.") 
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

        if connection == "tunnel":
            from pyngrok import ngrok
            for launcher in launchers:
                try:
                    ngrok.disconnect(launcher.url)
                except Exception:
                    pass
        
    elif hosting == "remote":
        typer.echo(
            "Remote simulation mode selected. This machine does not support launching remote simulations. "
            "Please run the simulation command directly on the simulator, which hosts the simulation locally."
        )
    elif hosting == "cloud":
        typer.echo("Cloud-based simulation is not supported yet.")
        raise typer.Exit(code=0)
    else:
        typer.echo("Other hosting modes are not supported yet.")

def initialize_sagemaker_access(
    role_arn: str,
    region: str,
    service_type: str,  # expected to be "trainer" or "inference"
    email: Optional[str] = None
):
    """
    Initialize SageMaker access by registering your AWS account details.

    - Validates the role ARN format.
    - Extracts your AWS account ID from the role ARN.
    - Sends the account ID, region, and service type to the registration endpoint.
    
    Returns True on success; otherwise, returns False.
    """
    # Validate the role ARN format.
    if not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+$", role_arn):
        typer.echo("Invalid role ARN format.")
        return False

    try:
        account_id = role_arn.split(":")[4]
    except IndexError:
        typer.echo("Invalid role ARN. Unable to extract account ID.")
        return False

    typer.echo("Initializing access...")
    
    beta_register_url = "https://agentgpt-beta.ccnets.org"
    payload = {
        "clientAccountId": account_id,
        "region": region,
        "serviceType": service_type
    }
    if email:
        payload["Email"] = email
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(beta_register_url, json=payload, headers=headers)
    except Exception:
        typer.echo("Request error.")
        return False

    if response.status_code != 200:
        typer.echo("Initialization failed.")
        return False

    if response.text.strip() in ("", "null"):
        typer.echo("Initialization succeeded.")
        return True

    try:
        data = response.json()
    except Exception:
        typer.echo("Initialization failed.")
        return False

    if data.get("statusCode") == 200:
        typer.echo("Initialization succeeded.")
        return True
    else:
        typer.echo("Initialization failed.")
        return False

@app.command(
    "train",
    short_help=help_texts["train"]["short_help"],
    help=auto_format_help(help_texts["train"]["detailed_help"])
)
def train():
    config_data = load_config()

    input_config_names = ["sagemaker", "hyperparams"] 
    input_config = {}
    for name in input_config_names:
        input_config[name] = config_data.get(name, {})
    converted_obj = convert_to_objects(input_config)
    
    sagemaker_obj: SageMakerConfig = converted_obj["sagemaker"]
    hyperparams_config: Hyperparameters = converted_obj["hyperparams"]
    
    if not initialize_sagemaker_access(sagemaker_obj.role_arn, sagemaker_obj.region, service_type="trainer"):
        typer.echo("AgentGPT training failed.")
        raise typer.Exit(code=1)
    
    typer.echo("Submitting training job...")
    estimator = AgentGPT.train(sagemaker_obj, hyperparams_config)
    typer.echo(f"Training job submitted: {estimator.latest_training_job.name}")

@app.command(
    "infer",
    short_help=help_texts["infer"]["short_help"],
    help=auto_format_help(help_texts["infer"]["detailed_help"])
)
def infer():
    config_data = load_config()

    # Use the sagemaker configuration.
    input_config_names = ["sagemaker"]
    input_config = {name: config_data.get(name, {}) for name in input_config_names}
    converted_obj = convert_to_objects(input_config)
    
    sagemaker_obj: SageMakerConfig = converted_obj["sagemaker"]

    if not initialize_sagemaker_access(sagemaker_obj.role_arn, sagemaker_obj.region, service_type="inference"):
        typer.echo("Error initializing SageMaker access for AgentGPT inference.")
        raise typer.Exit(code=1)

    typer.echo("Deploying inference endpoint...")
    
    gpt_api = AgentGPT.infer(sagemaker_obj)
    typer.echo(f"Inference endpoint deployed: {gpt_api.endpoint_name}")

@app.callback()
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("No command provided. Displaying help information:\n")
        typer.echo(ctx.get_help())
        raise typer.Exit()

if __name__ == "__main__":
    app()   
