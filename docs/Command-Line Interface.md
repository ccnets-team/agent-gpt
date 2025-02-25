
# Command-Line Interface

The AgentGPT CLI provides a set of commands that allow you to interact with the application directly from your terminal.

## Commands

### Config
Update configuration settings. This command accepts extra options (e.g., `--batch_size`, `--role_arn`, `--set_env_host`, etc.) and saves the result so that subsequent `agent-gpt train` or `agent-gpt infer` commands use the updated configuration.

```bash
# View current configuration
agent-gpt config

# Update configuration with hyperparameters and SageMaker settings
agent-gpt config --batch_size 256 --role_arn arn:aws:iam::123456789012:role/AgentGPTSageMakerRole
agent-gpt config --set_env_host cloud1 your_endpoint_on_cloud 32
```

The same `agent-gpt config` command lists the full stored configuration (i.e., defaults merged with any overrides).

### Clear
Delete the configuration cache and reset the CLI state.

```bash
agent-gpt clear
```

### Simulate
Launch simulation environments. This command supports different modes:
- **Local Simulation:** Provide port values (or your IP is auto-detected, so you only need to supply `--port` info) to launch local simulators.
- **Cloud Simulation:** For now, launch your environment on the cloud using your own infrastructure (e.g., EKS or EC2) but be sure to configure them in `env_hosts` before triggering a SageMaker training job.

```bash
# Launch a local gym simulation on specified ports
agent-gpt simulate gym port1 port2 ...
```

### Train
Initiate a training job on AWS SageMaker. This command loads configuration settings (such as hyperparameters and data paths) and submits a training job to the cloud.

```bash
agent-gpt train
```

### Infer
Deploy or reuse a SageMaker inference endpoint using the stored configuration settings. This command runs the agents on AWS to serve the trained model.

```bash
agent-gpt infer
```

## How It Works

Each CLI command is defined using a decorator (e.g., `@app.command()`) from a CLI framework like Typer. This approach:
- Exposes functions as terminal commands.
- Automatically generates help messages.
- Manages command-line parsing.

## Summary

The AgentGPT CLI enables structured interaction with the application by letting you specify actions and provide necessary options or arguments. Whether you're updating configurations, launching simulations, training models, or running inference, these commands streamline your workflow.

For more details on each command or advanced options, you can always run:
```bash
agent-gpt --help
```
```

Simply copy and paste the above content into your `docs/Command-Line Interface.md` file.
