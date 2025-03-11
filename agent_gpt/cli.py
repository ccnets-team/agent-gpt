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
import json
import websocket
from .core import AgentGPT
from .config.sagemaker import SageMakerConfig
from .config.hyperparams import Hyperparameters
from typing import Optional, Dict
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

# Define a function to poll for config changes.
def wait_for_config_update(sent_identifier, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        config_data = load_config()  # Your function to load the config file.
        checked_identifier = config_data.get("hyperparams", {}).get("training_key")
        # Check if all expected keys are present.
        if sent_identifier == checked_identifier:
            return config_data
        time.sleep(0.5)
    raise TimeoutError("Timed out waiting for config update.")

def envelop_request(action, data):
    return json.dumps({"action": action, "data": data})

def get_agent_gpt_server_url(region):
    # check if the region is valid
    if region not in ["ap-northeast-2", "us-east-1", "us-west-2", "eu-west-1"]:
        raise ValueError(f"Invalid region: {region}")
    return f"wss://{region}.agentgpt-beta.ccnets.org"

# Allow the user not to provide the arguments upfront.
@app.command("simulate")
def simulate(
    env_type: Optional[str] = typer.Option(None, help="Environment type: 'gym' or 'unity'"),
    env_id: Optional[str] = typer.Option(None, help="Environment ID to simulate, e.g., 'Walker2d-v5'"),
    num_envs: Optional[int] = typer.Option(None, help="Number of parallel environments to simulate concurrently"),
    num_agents: Optional[int] = typer.Option(None, help="Number of agents to simulate and train"),
    region: Optional[str] = typer.Option(None, help="Your region for running simulation/training"),
    entry_point: Optional[str] = typer.Option(None, help="Entry point script for the simulation"),
    env_dir: Optional[str] = typer.Option(None, help="Directory containing the simulation environment files"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    if not env_type:
        env_type = typer.prompt(
            "Please provide the environment type ('gym' or 'unity')"
        )

    if not env_id:
        env_id = typer.prompt(
            "Please provide the environment ID to simulate (e.g., 'Walker2d-v5')"
        )

    if not num_agents:
        num_agents = typer.prompt(
            "Please provide the number of agents to simulate and train", type=int
        )

    if not num_envs:
        num_envs = typer.prompt(
            "Please provide the number of parallel environments", type=int
        )

    if not region:
        region = typer.prompt(
            "Please provide the AWS region for conneting your local environment ? training (e.g., 'ap-northeast-2')",
            default="ap-northeast-2",
        )
        
    env_config = {
        "env_type": env_type,
        "env_id": env_id,
        "num_envs": num_envs,
        "num_agents": num_agents,
        "entry_point": entry_point,
        "env_dir": env_dir,
        "seed": seed
    }

    ws = websocket.WebSocket()
    agent_gpt_server_url = get_agent_gpt_server_url(region)
    ws.connect(agent_gpt_server_url)
    message = envelop_request("register", env_config)
    ws.send(message)
    remote_training_key = ws.recv()
    ws.close()
    
    # Start with your env_config dict
    extra_args = []

    # Add remote_training_key explicitly first
    extra_args.extend(["--agent_gpt_server_url", agent_gpt_server_url])
    extra_args.extend(["--remote_training_key", remote_training_key])

    # Dynamically add all other args from env_config
    for key, value in env_config.items():
        # Skip None values (optional)
        if value is not None:
            extra_args.extend([f"--{key}", str(value)])

    typer.echo("Starting the simulation in a separate terminal window. Please monitor that window for real-time logs.")
    
    from .simulation import open_simulation_in_screen
    simulation_process = open_simulation_in_screen(extra_args)
    try:
        updated_config = wait_for_config_update(remote_training_key, timeout=10)
        remote_training_key = updated_config.get("hyperparams", {}).get("remote_training_key", {})
        typer.echo("Remote Training Key for simulation updated successfully:")
        dislay_output = "**hyperparams**:\n" + yaml.dump(remote_training_key, default_flow_style=False, sort_keys=False)
        typer.echo(typer.style(dislay_output.strip(), fg=typer.colors.GREEN))
    except TimeoutError:
        typer.echo("Configuration update timed out. Terminating simulation process.")
        simulation_process.terminate()
        simulation_process.wait()
    
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
