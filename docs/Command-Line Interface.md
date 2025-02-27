
# Command-Line Interface

The AgentGPT CLI provides a set of commands that allow you to interact with the application directly from your terminal.

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
Update configuration settings. This command accepts various options (e.g., --batch_size, --role_arn, --set_env_host, etc.) and saves the results so that subsequent agent-gpt train or agent-gpt infer commands use the updated configuration.

- **Basic Configuration:**  
  Update global settings such as hyperparameters and AWS credentials.
  ```bash
  # Update global configuration with hyperparameters and SageMaker settings
  agent-gpt config --batch_size 256
  agent-gpt config --region us-east-1 --role_arn arn:aws:iam::123456789012:role/AgentGPTSageMakerRole
  ```

- **Module Configuration:**  
  Manage specific modules like environment hosts and exploration methods, including adding or removing them.
  ```bash
  # Add a cloud environment host with 32 agents
  agent-gpt config --set_env_host endpoint_name your_endpoint_on_cloud 32
  ```

  ```bash
  # Remove an environment host configuration
  agent-gpt config --del_env_host endpoint_name
  ```

  ```bash
  # Add an exploration method for continuous control using gaussian_noise
  agent-gpt config --set_exploration continuous gaussian_noise
  agent-gpt config --set_exploration discrete epsilon_greedy
  ```

  ```bash
  # Remove an exploration method configuration
  agent-gpt config --del_exploration discrete
  ```

- **Nested Configuration:**  
  Update deeply nested configuration parameters using dot notation for fine-grained control.
  ```bash
  # Change the initial sigma for continuous exploration
  agent-gpt config --exploration.continuous.initial_sigma 0.2 
  ```

  ```bash
  # Set the number of agents for a specific environment host
  agent-gpt config --env_hosts.cloud1.num_agents 512
  ```

### List
View current configuration.
```bash
agent-gpt list
```
Output example:
```bash
Current configuration:
environment:
  envs:
    cloud1:
      dockerfile:
        additional_dependencies: []
      entry_point: null
      env: gym
      env_id: null
      env_path: ''
      host_name: https://ABCDEF0123456789.gr7.eks.us-east-1.amazonaws.com
      host_type: cloud
      k8s_manifest:
        deployment_name: agent-gpt-cloud-env-k8s
        image_name: ''
      ports:
      - 55670
      - 55671
      - 55672
      - 55673
    local:
      dockerfile:
        additional_dependencies: []
      entry_point: null
      env: gym
      env_id: null
      env_path: ''
      host_name: http://198.51.100.23
      host_type: local
      k8s_manifest:
        deployment_name: agent-gpt-cloud-env-k8s
        image_name: ''
      ports:
      - 45670
      - 45671
      - 45672
      - 45673
hyperparams:
  batch_size: 256
  buffer_size: 1000000
  d_model: 256
  dropout: 0.1
  env_hosts:
    cloud1_0:
      env_endpoint: https://ABCDEF0123456789.gr7.eks.us-east-1.amazonaws.com:55670
      num_agents: 32
    cloud1_1:
      env_endpoint: https://ABCDEF0123456789.gr7.eks.us-east-1.amazonaws.com:55671
      num_agents: 64
    local_0:
      env_endpoint: http://198.51.100.23:45670
      num_agents: 128
  env_id: null
  exploration:
    continuous:
      dt: null
      final_epsilon: null
      final_sigma: 0.001
      final_stddev: null
      initial_epsilon: null
      initial_sigma: 0.1
      initial_stddev: null
      mu: null
      ou_sigma: null
      theta: null
      type: gaussian_noise
  gamma_init: 0.99
  gpt_type: gpt2
  lambda_init: 0.95
  lr_end: 1.0e-05
  lr_init: 0.0001
  lr_scheduler: linear
  max_grad_norm: 0.5
  max_input_states: 16
  max_steps: 20000000
  num_heads: 8
  num_layers: 5
  replay_ratio: 2.0
  resume_training: false
  tau: 0.01
  use_cloudwatch: true
  use_graphics: false
  use_tensorboard: false
network:
  host: 0.0.0.0
  internal_ip: 328.54.120.28
  public_ip: 198.51.100.23
sagemaker:
  inference:
    endpoint_name: agent-gpt-inference-endpoint
    image_uri: 533267316703.dkr.ecr.ap-northeast-2.amazonaws.com/agent-gpt-inference:latest
    instance_count: 1
    instance_type: ml.t2.medium
    max_run: 3600
    model_data: s3://your-bucket/model.tar.gz
  region: ap-northeast-2
  role_arn: arn:aws:iam::<your-account-id>:role/SageMakerExecutionRole
  trainer:
    image_uri: 533267316703.dkr.ecr.ap-northeast-2.amazonaws.com/agent-gpt-trainer:latest
    instance_count: 1
    instance_type: ml.g5.4xlarge
    max_run: 3600
    output_path: s3://your-bucket/output/
```

### Clear
Reset the configuration cache and reset the CLI state.
```bash
agent-gpt clear
```

### Simulate
Launch simulation environments. This command supports different modes:
- **Local Simulation:** Provide port values (or your IP is auto-detected, so you only need to supply `--port` info) to launch local simulators.
- **Cloud Simulation:** For now, launch your environment on the cloud using your own infrastructure (e.g., EKS or EC2) but be sure to configure them in `env_hosts` before triggering a SageMaker training job.

```bash
# Launch a local gym simulation on specified ports
agent-gpt simulate local port1 port2 ...
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

```

Simply copy and paste the above content into your `docs/Command-Line Interface.md` file.
