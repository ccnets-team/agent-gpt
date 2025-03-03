# Command-Line Interface

The AgentGPT CLI provides a set of commands to interact with the application directly from your terminal. It makes it easy to configure, simulate, train, and deploy your multi‑agent reinforcement learning environments.

## Commands

### Help
Display help information and usage guidelines for AgentGPT CLI commands.
```bash
agent-gpt --help
agent-gpt config --help
agent-gpt simulate --help
...
```

### Config
Update configuration settings that are used by subsequent training or inference commands.

**Basic Configuration:**  
Update global settings such as hyperparameters and AWS Sagemaker setting.
```bash
# Update global configuration with hyperparameters and SageMaker settings
agent-gpt config --batch_size 256
agent-gpt config --region us-east-1 --role_arn arn:aws:iam::123456789012:role/AgentGPTSageMakerRole
```

**Advanced Module Configuration:**  
For users who wish to fine-tune or override settings—such as environment hosts, exploration methods, or simulator registry details—the CLI also supports advanced commands.  
> **Note:**  
> In most cases, the simulation command automatically configures environment hosts for you. Advanced users can manually adjust these settings if needed.

```bash
# Manually add or remove an environment host configuration.
agent-gpt config env_host set endpoint_name --env_endpoint your_endpoint_on_cloud --num_agents 32
agent-gpt config env_host del endpoint_name
```

```bash
# Configure exploration methods for continuous or discrete control.
agent-gpt config exploration set continuous --type gaussian_noise
agent-gpt config exploration set discrete --type epsilon_greedy
agent-gpt config exploration del discrete
```

**Nested Configuration:**  
Update deeply nested configuration parameters using dot notation for fine-grained control.
```bash
# Change the initial sigma for continuous exploration
agent-gpt config --exploration.continuous.initial_sigma 0.2 

# Set the number of agents for a specific environment host (advanced use-case)
agent-gpt config --env_host.cloud1.num_agents 512
```

### List
View the current configuration.
```bash
agent-gpt list
```
Example output:
```yaml
**simulator_registry**:
simulators:
  local:
    env_type: gym
    hosting: local
    url: http://203.0.113.57
    host: 0.0.0.0
    connection: tunnel
    total_agents: 256
    env_dir: /home/ccnets/Projects/agent-gpt
    ports:
      - 34560
      - 34561
      - 34562
      - 34563
    container:
      deployment_name: null
      image_uri: null
      additional_dependencies: []
      
**network**:
host: 0.0.0.0
public_ip: 203.0.113.57
internal_ip: NEW_INTERNAL_IP

**hyperparams**:
env_id: Walker2d-v5
env_entry_point: null
env_dir: null
env_host:
  local:34560:
    env_endpoint: https://106c-203-0-113-57.ngrok-free.app:34560
    num_agents: 64
  local:34561:
    env_endpoint: https://106c-203-0-113-57.ngrok-free.app:34561
    num_agents: 64
  local:34562:
    env_endpoint: https://106c-203-0-113-57.ngrok-free.app:34562
    num_agents: 64
  local:34563:
    env_endpoint: https://106c-203-0-113-57.ngrok-free.app:34563
    num_agents: 64
...
```

### Clear
Reset the configuration cache and CLI state.
```bash
agent-gpt clear
```

### Simulate
Launch simulation environments.  
When you run the `simulate` command, it automatically configures environment hosts for you—so you can immediately see the simulation status streaming in your terminal.
- **Local Simulation:**  
  Launch local simulators (the CLI automatically sets up environment host details).
  
- **Cloud Simulation:**  
  For now, use your own infrastructure (e.g., EKS, EC2, or App Runner) with connection modes such as:
  - **Local:** Using direct IP or a tunnel (e.g., via ngrok).
  - **Cloud:** Connecting via EC2, EKS, or App Runner.
  
```bash
# Launch a local gym simulation
agent-gpt simulate local
```

> **Note:**  
> When running `simulate simulator_id`, the CLI will block the terminal to stream the simulation status. This is expected behavior, as it allows you to monitor the simulation in real time.

### Train
Initiate a training job on AWS SageMaker.  
This command uses your configuration (including hyperparameters and sagemaker configurations) to submit a training job to the cloud.
```bash
agent-gpt train
```

### Infer
Deploy or reuse a SageMaker inference endpoint using your stored configuration.  
This command deploys your trained model so that agents can run on AWS.
```bash
agent-gpt infer
```

### Simulator Registry
The Simulator Registry is an innovative feature that manages multiple simulators—whether local or remote—and allows you to control which simulators participate in your next training job. Key benefits include:

- **Managing Multiple Simulators:**  
  Easily add, remove, and list simulator configurations. This enables you to select specific simulators for large-scale, controlled cloud training sessions.

- **Dynamic, Automatic Configuration:**  
  When you launch a simulation using agent-gpt simulate simulator_id, the CLI automatically updates the simulator registry with the appropriate environment host information for that specific simulator. This minimizes manual configuration while allowing you to manage and control individual simulators for a streamlined training setup.

- **Customization with methods:**  
  You can manually adjust or override simulator configurations via commands such as:
  ```bash
  # Add or update a simulator configuration.
  agent-gpt config simulator set my_remote_simulator --env_type gym --hosting cloud --url http://your_remote_endpoint
  
  # Remove a simulator from the registry.
  agent-gpt config simulator del my_remote_simulator 
  ```

> **Note:**  
> For most users, simply launching a simulation will auto-update the simulator registry. The advanced registry commands are available for users who need extra customization or integration with a future graphical interface.

## How It Works

Each CLI command is defined using decorators (e.g., `@app.command()`) from a CLI framework like Typer. This approach:
- Exposes functions as terminal commands.
- Automatically generates help messages.
- Manages command-line parsing.

## Summary

The AgentGPT CLI streamlines your workflow by providing commands to update configurations, launch simulations, train models, and deploy inference endpoints. Key innovations include:

- **Automatic Environment Host Configuration:**  
  When you run a specific simulator with `simulate` command, the environment host settings are auto-configured, so you don’t need to manually adjust them.

- **Simulator Registry:**  
  This feature manages multiple simulators (local and remote), enabling massive yet controlled cloud training sessions through AWS SageMaker.

- **Flexible Connection Modes:**  
  Whether you’re using a local IP, a tunnel, or a cloud-based connection (via EC2, EKS, or App Runner), AgentGPT supports a range of connection methods to suit your deployment needs.

---
