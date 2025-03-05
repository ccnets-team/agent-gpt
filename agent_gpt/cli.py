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
import time
import yaml
import requests
from typing import Optional, List, Dict
from .config.simulator import SimulatorConfig 
from .config.hyperparams import Hyperparameters
from .config.sagemaker import SageMakerConfig
from .core import AgentGPT
from .utils.config_utils import load_config, save_config, generate_default_section_config, update_config_using_method, ensure_config_exists
from .utils.config_utils import convert_to_objects, parse_extra_args, update_config_by_dot_notation
from .utils.config_utils import DEFAULT_CONFIG_PATH, TOP_CONFIG_CLASS_MAP

app = typer.Typer(add_completion=False, invoke_without_command=True)

def load_help_texts(yaml_filename: str) -> Dict:
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
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    short_help=help_texts["config"]["short_help"],
    help=auto_format_help(help_texts["config"]["detailed_help"]),
)
def config(ctx: typer.Context):
    ensure_config_exists()
    
    if not ctx.args:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    
    current_config = load_config()
    config_obj = convert_to_objects(current_config)

    # Decide the mode based on the first argument.
    if ctx.args[0].startswith("--"):
        new_changes = parse_extra_args(ctx.args)
        update_log = update_config_by_dot_notation(config_obj, new_changes)
    else:
        update_log = update_config_using_method(ctx.args, config_obj)

    # Print detailed change summaries.
    for key, old_value, new_value, changed, message in update_log:
        if changed:
            method_configuration = True if old_value is None and new_value is None else False
            if method_configuration:
                typer.echo(typer.style(
                    f" - {key} {message}.",
                    fg=typer.colors.GREEN
                ))
            else:
                typer.echo(typer.style(
                    f" - {key} changed from {old_value} to {new_value}",
                    fg=typer.colors.GREEN
                ))
        else:
            typer.echo(typer.style(
                f" - {key}: no changes applied because {message}",
                fg=typer.colors.YELLOW
            ))
            
    full_config = {key: obj.to_dict() for key, obj in config_obj.items()}
    save_config(full_config)

@app.command(
    "edit",
    short_help=help_texts["edit"]["short_help"],
    help=auto_format_help(help_texts["edit"]["detailed_help"]),
)
def edit_config():   
    ensure_config_exists()
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
        typer.echo(typer.style(f"Failed to open the configuration file: {e}", fg=typer.colors.YELLOW))

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
    ensure_config_exists()
    
    allowed_sections = set(TOP_CONFIG_CLASS_MAP.keys())
    if section:
        if section not in allowed_sections:
            typer.echo(typer.style(f"Invalid section '{section}'. Allowed sections: {', '.join(allowed_sections)}.", fg=typer.colors.YELLOW))
            raise typer.Exit()
        current_config = load_config()
        current_config[section] = generate_default_section_config(section)
        save_config(current_config)
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
    ensure_config_exists()
    
    current_config = load_config()
    if section in TOP_CONFIG_CLASS_MAP.keys():
        # Retrieve the specified section and print its contents directly.
        if section not in current_config:
            typer.echo(f"No configuration found for section '{section}'.")
            return
        typer.echo(f"Current configuration for '{section}':")
        typer.echo(yaml.dump(current_config[section], default_flow_style=False, sort_keys=False))
    else:
        typer.echo("Current configuration:")
        for sec in TOP_CONFIG_CLASS_MAP.keys():
            if sec in current_config:
                typer.echo(f"**{sec}**:")
                typer.echo(yaml.dump(current_config[sec], default_flow_style=False, sort_keys=False))

@app.command("upload",
             short_help=help_texts["upload"]["short_help"],
             help=auto_format_help(help_texts["upload"]["detailed_help"]))
def upload(simulator_id: str = typer.Argument()):
    ensure_config_exists()
    
    current_data = load_config()
    
    sagemaker_config = current_data.get("sagemaker", {})
    region = sagemaker_config.get("region")
    if not region:
        typer.echo(typer.style("Error: AWS region not set in the configuration.", fg=typer.colors.YELLOW))
        raise typer.Exit(code=1)

    registry_data = current_data.get("simulator_registry", {})
    simulator_group_data = registry_data.get("simulator", {})
    simulator_data = simulator_group_data.get(simulator_id, {})
    simulator = SimulatorConfig()
    simulator.set_config(**simulator_data)
    
    if simulator.hosting != "cloud":
        typer.echo(typer.style(f"Error: Simulator '{simulator_id}' is not set up for cloud deployment.", fg=typer.colors.YELLOW))
        raise typer.Exit(code=1)

    from .env_host.upload import upload_simulator
    try: 
        upload_simulator(region, simulator)
        typer.echo(f"Simulator '{simulator_id}' uploaded successfully.")
    except Exception as e:
        typer.echo(typer.style(f"Error uploading simulator '{simulator_id}': {e}", fg=typer.colors.YELLOW))
        raise typer.Exit(code=1)

    registry = current_data.setdefault("simulator_registry", {})
    simulators = registry.setdefault("simulator", {})
    simulators[simulator_id] = simulator.to_dict()
    save_config(current_data)

# Define a function to poll for config changes.
def wait_for_config_update(expected_keys, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        config_data = load_config()  # Your function to load the config file.
        env_host_keys = config_data.get("hyperparams", {}).get("env_host", {})
        # Check if all expected keys are present.
        if all(key in env_host_keys for key in expected_keys):
            return config_data
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for config update.")

@app.command(
    "simulate",
    short_help=help_texts["simulate"]["short_help"],
    help=auto_format_help(help_texts["simulate"]["detailed_help"])
)
def simulate(
    simulator_id: str = typer.Argument(None, help="Simulator identifier"),
    ports: List[int] = typer.Argument(None, help="List of port numbers")
):
    ensure_config_exists()
    
    # Load configuration and simulator data.
    current_data = load_config()
    simulator_registry_data = current_data.get("simulator_registry", {})
    simulator = simulator_registry_data.get("simulator", {})

    simulator_identifiers = list(simulator.keys())
    num_simulators = len(simulator_identifiers)

    if simulator_id and simulator_id not in simulator_identifiers:
        typer.echo(typer.style(f"Error: No simulator found with identifier '{simulator_id}'.", fg=typer.colors.YELLOW))
        simulator_id = None

    if not simulator_id and num_simulators > 0:
        typer.echo("Available simulator identifiers:")
        for sid in simulator_identifiers:
            typer.echo(f"  - {sid}")
        simulator_id = typer.prompt(
            "Enter the simulator identifier to use",
            default=simulator_identifiers[0],
            show_default=True
        )
   
    if not simulator_id:
        raise typer.Exit(code=1)

    simulator_data = simulator.get(simulator_id)
    
    if not simulator_data:
        typer.echo(typer.style(f"Error: No simulator found with identifier '{simulator_id}'.", fg=typer.colors.YELLOW))
        raise typer.Exit(code=1)
    
    hosting = simulator_data.get("hosting")
    if hosting != "local":
        typer.echo(typer.style(
                "Direct simulation control is available only for local simulators. "
                "If your simulation is launched externally, you can still manually specify the env_endpoint "
                "and update your hyperparameters before submitting a training job (using the 'agent-gpt train' command).",
                fg=typer.colors.YELLOW))
        raise typer.Exit(code=1)
    
    # Check for ports.
    if not ports:
        typer.echo("No port numbers provided. Using ports from configuration.")
        ports = simulator_data.get("ports", [])
    if not ports:
        typer.echo(typer.style("Error: No available ports found. Please specify one or more port numbers.", fg=typer.colors.YELLOW))
        raise typer.Exit(code=1)

    # Prepare the extra arguments to pass to simulation.py.
    # (Ports are passed as a comma-separated string.)
    port_arg = ",".join(str(p) for p in ports)
    extra_args = [
        "--simulator_id", simulator_id,
        "--ports", port_arg,
        "--env_type", simulator_data.get("env_type", ""),
        "--connection", simulator_data.get("connection", ""),
        "--host", simulator_data.get("host", "0.0.0.0"),
        "--total_agents", str(simulator_data.get("total_agents", 128)),
        "--url", simulator_data.get("url", "")
    ]
    typer.echo("Starting the simulation in a separate terminal window. Please monitor that window for real-time logs.")
    
    # Launch the new process that will execute the simulation logic.
    from .simulation import open_simulation_in_screen
    simulation_process = open_simulation_in_screen(extra_args)
    expected_keys = [f"{simulator_id}:{port}" for port in ports]
    try:
        updated_config = wait_for_config_update(expected_keys, timeout=10)
        updated_hyperparms = updated_config.get("hyperparams", {})
        env_host = updated_hyperparms.get("env_host", {})
        # print("Configuration updated:", updated_config["hyperparams"])
        typer.echo(f"Environment hosts for simulation '{simulator_id}' have been updated successfully:")
        typer.echo("Below is the updated configuration for environment hosts:")
        typer.echo("Hyperparameters have been auto-configured for cloud training.")
        dislay_output = "**hyperparams**:\n" + yaml.dump(env_host, default_flow_style=False, sort_keys=False)
        typer.echo(typer.style(dislay_output.strip(), fg=typer.colors.GREEN))        
        typer.echo("Simulation has been launched. You may continue to work in this terminal for further commands or to initiate another simulation.")
        
    except TimeoutError as e:
        typer.echo("Configuration update timed out. The simulation process will now be forcefully terminated to free up resources.")
        simulation_process.terminate()  # This call is non-blocking.
        simulation_process.wait()
    
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
        typer.echo(typer.style("Invalid role ARN format.", fg=typer.colors.YELLOW))
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
        typer.echo(typer.style("Initialization failed.", fg=typer.colors.YELLOW))
        return False

    if response.text.strip() in ("", "null"):
        typer.echo("Initialization succeeded.")
        return True

    try:
        data = response.json()
    except Exception:
        typer.echo(typer.style("Initialization failed.", fg=typer.colors.YELLOW))
        return False

    if data.get("statusCode") == 200:
        typer.echo("Initialization succeeded.")
        return True
    else:
        typer.echo(typer.style("Initialization failed.", fg=typer.colors.YELLOW))
        return False

@app.command(
    "train",
    short_help=help_texts["train"]["short_help"],
    help=auto_format_help(help_texts["train"]["detailed_help"])
)
def train():
    ensure_config_exists()
    
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
    ensure_config_exists()
    
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
