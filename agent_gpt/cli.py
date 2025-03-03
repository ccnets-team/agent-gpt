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
import textwrap
import os
import re
import yaml
import requests
from typing import Optional, List
from .config.simulator import SimulatorConfig 
from .config.hyperparams import Hyperparameters
from .config.sagemaker import SageMakerConfig
from .env_host.server import EnvServer
from .core import AgentGPT
from .utils.config_utils import load_config, save_config, generate_section_config, handle_config_method
from .utils.config_utils import convert_to_objects, parse_extra_args, initialize_config, apply_config_updates
from .utils.config_utils import DEFAULT_CONFIG_PATH, TOP_CONFIG_CLASS_MAP

app = typer.Typer(add_completion=False, invoke_without_command=True)

@app.command(
    "config",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def config(ctx: typer.Context):
    """
    Update configuration settings.\n\n
    
    This command supports two modes for modifying configuration: \n\n
    
    1. Field Update Mode: \n
    Use dot notation to update configuration fields directly.\n\n
    
    For example: \n
    agent-gpt config --batch_size 64 --lr_init 0.0005 --env_id CartPole-v1 \n
    agent-gpt config --trainer.max_run 360 \n\n
    
    This mode updates any key in the top-level configuration (such as:
    simulator_registry, network, hyperparams, sagemaker)
    without requiring dedicated subcommands. \n\n

    2. Method Mode: \n
    Use dedicated methods to add or remove configuration entries in specific sections. \n
    The following functions are available, using the syntax:\n
    agent-gpt config simulator/env-host/exploration set/del [identifier] [--option value ...] \n

    a. Simulator Configuration:\n\n
    
    - Set:\n
    agent-gpt config simulator set my_simulator --hosting local --connection ip\n
    agent-gpt config simulator set my_simulator --hosting cloud --connection ec2\n
    - Delete:\n
    agent-gpt config simulator del my_simulator\n\n
    
    b. Environment Host Configuration:\n\n
    
    - Set:\n
    agent-gpt config env-host set local0 --env_endpoint http://your-host:port --num_agents 32\n
    agent-gpt config env-host set local1 --env_endpoint http://your-host:port1 --num_agents 64\n
    - Delete:\n
    agent-gpt config env-host del local0\n
    
    c. Exploration Configuration:\n\n
    
    - Set:\n
    agent-gpt config exploration set continuous --type gaussian_noise --param1 0.1 --param2 0.001\n
    - Delete:\n
    agent-gpt config exploration del continuous\n\n

    Choose Field Update Mode for simple, direct key modifications and Method Mode for more guided, complex configuration changes.
    """
    
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

@app.command("edit")
def edit_config():
    """
    Open the configuration file in the system's default text editor for manual modification.
    If the configuration file does not exist, create one with default values.
    """
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

@app.command("upload")
def upload(
    simulator_id: str = typer.Argument(
        help="Simulator ID to upload for cloud hosting."
    )
):
    """
    Upload a simulator for cloud deployment.

    Steps:
      1. Validate & Retrieve Simulator:
         - Load the configuration and retrieve the simulator settings from the Simulator Registry.
      2. Create Dockerfile & Upload:
         - Generate a Dockerfile based on the simulator configuration.
         - Build and push a Docker image to your ECR account.
      3. Update Simulator Registry:
         - Update the Simulator configuration with the new image URI after a successful upload.
    
    Example:
      agent-gpt upload my_simulator
    """
    config_data = load_config()
    region = config_data.get("sagemaker", {}).get("region")
    if not region:
        typer.echo("Error: AWS region not set in the configuration.")
        raise typer.Exit(code=1)

    simulator_registry_data = config_data.get("simulator_registry", {})
    simulators = simulator_registry_data.get("simulators", {})
    simulator_data = simulators.get(simulator_id)
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
    
    simulator_registry_data["simulators"][simulator_id] = simulator_config.to_dict()
    config_data["simulator_registry"] = simulator_registry_data
    save_config(config_data)

@app.command("simulate")
def simulate(
    simulator_id: str = typer.Argument(
        "local",
        help="Environment identifier to simulate. Default: 'local'."
    ),
    ports: Optional[List[int]] = typer.Argument(
        None,
        help="One or more container port numbers on which to run the simulation server. Example: 80. If not provided, the simulator's default ports will be used."
    )
):
    """
    Launch an environment simulation using the configured simulator settings or specified port numbers.

    Steps:
      1. Retrieve Simulator Configuration:
           - Load the simulator settings from the local configuration file.
           - Use default ports from the configuration if no port numbers are provided.
      2. Launch Simulation Server:
           - Start a simulation server on each provided port based on the simulator's hosting type.
             * Local: Runs the simulation server locally.
             * Remote: Not supported on this machine; run the simulation directly on the remote simulator.
             * Cloud: Cloud-based simulation is not supported yet.
      3. Monitor & Terminate:
           - The simulation runs in the current terminal.
           - Press Ctrl+C to gracefully terminate the simulation.

    Examples:
      agent-gpt simulate local
      agent-gpt simulate local 8080, 8081
      agent-gpt simulate my_simulator 80, 81, 82, 83
    """
    # Load configuration to get the network settings.
    config_data = load_config()

    simulator_registry_data = config_data.get("simulator_registry", {})
    simulators = simulator_registry_data.get("simulators", {})
    simulator_data = simulators.get(simulator_id)
    
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
        if connection == "tunnel":
            from .utils.tunnel import create_tunnel
            url = create_tunnel(ports)
            
        launchers = []
        for port in ports:
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
        env_hosts = config_data.get("hyperparams", {}).get("env_hosts", {})
        added_env_hosts = []
        # Store simulation host info using a key like f"{simulator_id}:{port}"
        for i, launcher in enumerate(launchers):
            key = f"{simulator_id}:{launcher.port}"
            env_endpoint = launcher.endpoint
            env_hosts[key] = {"env_endpoint": env_endpoint, "num_agents": agents_array[i]}
            added_env_hosts.append(key)
            typer.echo(f"env_endpoint: {env_endpoint}, num_agents: {agents_array[i]}")

        # Update and save the config.
        config_data.setdefault("hyperparams", {})["env_hosts"] = env_hosts
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

        # After simulation ends, remove only the env_hosts entries added for this simulation.
        for key in added_env_hosts:
            env_hosts.pop(key, None)

        config_data["hyperparams"]["env_hosts"] = env_hosts
        save_config(config_data)

        if connection == "tunnel":
            from pyngrok import ngrok
            ngrok.disconnect(url)
        
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

@app.command()
def train():
    """
    Launch a SageMaker training job for AgentGPT using configuration settings.
    This command loads training configuration from the saved config file.
    """
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

@app.command()
def infer():
    """
    Deploy or reuse a SageMaker inference endpoint for AgentGPT using configuration settings.
    This command loads inference configuration from the saved config file.
    """
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
