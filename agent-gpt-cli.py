#!/usr/bin/env python
import os
import yaml
import typer
from src.agent_gpt import AgentGPT
from src.config.sagemaker_config import SageMakerConfig
from src.config.hyperparams import Hyperparameters

app = typer.Typer()

# Default configuration file location (user-specific)
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.agent_gpt/config.yaml")

def load_config():
    if os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    else:
        typer.echo("Configuration file not found. Please run 'agent-gpt config' to create one.")
        raise typer.Exit()

@app.command()
def simulate(
    ip: str = typer.Option(..., help="The public IP address to launch the environment on"),
    port: str = typer.Option(..., help="Port or port range (e.g., 45670 or 45670:45673)"),
    env_simulator: str = typer.Option("gym", help="The environment simulator (gym, unity, custom)")
):
    """
    Launch a local environment using the specified IP and port or port range.
    """
    # Parse the port option. If a colon is found, treat it as a range.
    if ":" in port:
        parts = port.split(":")
        start_port = int(parts[0])
        end_port = int(parts[1])
        ports = list(range(start_port, end_port + 1))
    else:
        ports = [int(port)]
    
    from src.env_host.simulator import EnvSimulator
    launchers = []
    for p in ports:
        launcher = EnvSimulator.launch_on_local_with_ip(env_simulator=env_simulator, ip_address=ip, host=ip, port=p)
        launchers.append(launcher)
        typer.echo(f"Local environment hosted on http://{ip}:{p}")
    typer.echo(f"Launched {len(launchers)} local environment(s).")

@app.command()
def train(
    env_id: str = typer.Option(..., help="Identifier for the environment"),
    batch_size: int = typer.Option(256, help="Training batch size")
):
    """
    Launch a SageMaker training job for AgentGPT.
    """
    config = load_config()
    train_config = config.get("train", {})
    sagemaker_config = SageMakerConfig(
        role_arn=train_config.get("role_arn"),
        image_uri=train_config.get("image_uri"),
        model_data=train_config.get("model_data"),
        output_path=train_config.get("output_path"),
        instance_type=train_config.get("instance_type"),
        instance_count=train_config.get("instance_count"),
        region=train_config.get("region"),
        max_run=train_config.get("max_run")
    )
    hyperparams = Hyperparameters(env_id=env_id, batch_size=batch_size)
    typer.echo("Submitting training job...")
    # Assuming AgentGPT.train is the refactored method replacing train_on_cloud.
    estimator = AgentGPT.train(sagemaker_config, hyperparams)
    typer.echo(f"Training job submitted: {estimator.latest_training_job.name}")

@app.command()
def infer():
    """
    Deploy or reuse a SageMaker inference endpoint for AgentGPT.
    """
    config = load_config()
    infer_config = config.get("infer", {})
    sagemaker_config = SageMakerConfig(
        role_arn=infer_config.get("role_arn"),
        image_uri=infer_config.get("image_uri"),
        model_data=infer_config.get("model_data"),
        endpoint_name=infer_config.get("endpoint_name", "agent-gpt-inference-endpoint"),
        instance_type=infer_config.get("instance_type"),
        instance_count=infer_config.get("instance_count"),
        region=infer_config.get("region")
    )
    typer.echo("Deploying inference endpoint...")
    # Assuming AgentGPT.infer is the refactored method replacing run_on_cloud.
    gpt_api = AgentGPT.infer(sagemaker_config)
    typer.echo(f"Inference endpoint deployed: {gpt_api.endpoint_name}")

@app.command("config")
def config(
    role_arn: str = typer.Option(..., help="AWS IAM Role ARN for SageMaker"),
    image_uri: str = typer.Option("agentgpt.ccnets.org", help="ECR image URI"),
    model_data: str = typer.Option(..., help="S3 path to the model tarball"),
    output_path: str = typer.Option(..., help="S3 path for training outputs"),
    instance_type: str = typer.Option("ml.g5.4xlarge", help="SageMaker instance type"),
    instance_count: int = typer.Option(1, help="Number of instances"),
    region: str = typer.Option("ap-northeast-2", help="AWS region"),
    max_run: int = typer.Option(3600, help="Maximum training runtime in seconds")
):
    """
    Save SageMaker configuration to the default config file.
    """
    config = {
        "train": {
            "role_arn": role_arn,
            "image_uri": image_uri,
            "model_data": model_data,
            "output_path": output_path,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "region": region,
            "max_run": max_run
        },
        "infer": {
            "role_arn": role_arn,
            "image_uri": image_uri,
            "model_data": model_data,
            "endpoint_name": "agent-gpt-inference-endpoint",
            "instance_type": instance_type,
            "instance_count": instance_count,
            "region": region
        }
    }
    os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.dump(config, f)
    typer.echo(f"Configuration saved to {DEFAULT_CONFIG_PATH}")

@app.command("list")
def list_config():
    """
    List the current configuration and network information.
    """
    config = load_config()
    typer.echo("Current configuration:")
    typer.echo(yaml.dump(config))
    # Optionally, display network info.
    try:
        from src.utils.network_info import get_network_info
        network_info = get_network_info()
        typer.echo(f"Network info: {network_info}")
    except Exception as e:
        typer.echo(f"Could not retrieve network info: {e}")

if __name__ == "__main__":
    app()
