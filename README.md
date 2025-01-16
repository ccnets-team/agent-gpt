# AgentGPT: One-Click Cloud-Based Distributed RL

**Repository:** [ccnets-team/agent-gpt](https://github.com/ccnets-team/agent-gpt.git)  
**W&B Benchmarks:** [Weights & Biases Dashboard](https://wandb.ai/junhopark/agentgpt)

This README explains how AgentGPT orchestrates RL training on AWS SageMaker, hosts environments in the cloud or locally, and provides multi-agent GPT-based endpoints.

![How AgentGPT Works](https://imgur.com/r4hGxqO.png)
---

## 1. Overview
AgentGPT is a **one-click, cloud-based distributed RL** platform built for multi-agent environments running in **parallel**. With minimal setup, it can automatically package and host your environment (assigning it a URL or IP) or launch a training job on AWS SageMaker. A **GPT model** is then served in real time, supporting **multiple local or cloud simulators** connected to a single AgentGPT trainer in the cloud—scaling effortlessly from small proofs-of-concept to large-scale, asynchronous RL deployments.

> **Note for Large Enterprises**  
> You can **keep your actual environment simulations on local machines** (on-premise, behind the firewall, etc.) to maintain security and control data flow. Rather than running hundreds of cloud-hosted environments, you send environment states from your local servers to a **single lightweight trainer** in AWS. This architecture **drastically reduces hosting costs** and provides peace of mind for sensitive data.

**Key Features**:
- **Cloud Environment Hosting**: Automatic Docker packaging & SageMaker job submission.  
- **Parallel Environment Hosting**: Multiple environment endpoints (each at its own URL or IP) can run concurrently, whether on local machines or AWS EC2/ECR.  
- **Real-Time Inference**: GPT-based endpoints served via SageMaker.  
- **Distributed RL Agent Support**: Each environment endpoint feeds observations to—and receives actions from—a central policy, enabling fully distributed, scalable training.  

---
## 2. Architecture

1. **Environment Hosting**  
   - **Local**: A FastAPI server (optionally tunneled via ngrok/localtunnel).  
   - **Cloud**: Containerization via Docker, ECR, and EC2 to host your environments.

2. **AgentGPT**  
   - Coordinates AWS SageMaker training jobs (`train_on_cloud`).  
   - Exposes real-time inference endpoints (`run_on_cloud`).

3. **GPTAPI**  
   - Offers high-level methods for multi-agent actions and queries, e.g. `select_action`, `sample_action`, and `set_control_value`.

4. **Configs and Hyperparams**  
   - **`SageMakerConfig`**: AWS roles, Docker image URIs, and model artifact paths.  
   - **`Hyperparameters`**: RL/training fields such as environment IDs, batch sizes, exploration strategies, etc.

---

## 3. Major Components

### 3.1 `AgentGPT` (in `agent_gpt.py`)
Orchestrates cloud-based training and inference:
- **`train_on_cloud(sagemaker_config, hyperparameters)`**  
  - Validates configs, instantiates a `sagemaker.Estimator`, and launches a training job.  

- **`run_on_cloud(sagemaker_config, user_endpoint_name=None)`**  
  - Deploys or reuses a SageMaker real-time inference endpoint.  

### 3.2 `GPTAPI` (in `gpt_api.py`)
A high-level interface to the SageMaker endpoint:

- **`select_action(agent_ids, observations, terminated_agent_ids=None)`**  
  Returns a NumPy array of actions for each agent.

- **`set_num_input_states`** / **`get_num_input_states`**  
  Adjusts or retrieves the **input sequence length** for the GPT-based model. In other words, it controls how many past timesteps (context) the model sees when making action decisions.

- **`set_control_value`** / **`get_control_value`**  
  Adjusts or retrieves an agent’s performance scaling factor (`[0.0, 1.0]`), effectively empowering or limiting an agent’s abilities.

### 3.3 Configuration Objects

- **`SageMakerConfig`** (`config/aws_config.py`):  
  - Specifies AWS credentials, Docker URIs, instance types, S3 paths, etc.

- **`Hyperparameters`** (`config/hyperparams.py`):  
  - Defines RL training settings (environment ID, batch size, buffer size, max steps, etc.).

### 3.4 Environment Hosting

- **Local** (`env_host/local/`):  
  - A FastAPI-based `EnvAPI` server providing REST endpoints for `reset`, `step`, `action_space`, etc.  
  - `TunnelManager` can open tunnels (ngrok/localtunnel) for external access.

- **Cloud** (`env_host/cloud/`):  
  - `CloudEnvLauncher` automates Dockerfile generation, ECR pushes, and EC2 launches, so your environment is accessible publicly.

     ![Environment Launching Steps](https://imgur.com/awvPxyx.png)

---

## 4. Usage Flow

1. **Host Your Environment**  
   - Start it locally with `EnvLauncher.launch_on_local_with_url(...)`,  
   - Or deploy it on AWS (`EnvLauncher.launch_on_cloud(...)`).

2. **Define SageMakerConfig**  
   - Example:
     ```python
     sagemaker_cfg = SageMakerConfig(
         role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
         image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/agent-gpt:latest",
         model_data="s3://your-bucket/model.tar.gz"
     )
     ```
     
3. **Set Hyperparameters**  
   - Example:
     ```python
     hyperparams = Hyperparameters(
         env_id="CartPole-v1",
         batch_size=128,
         max_steps=1_000_000
     )
     ```

4. **Train on SageMaker**  
   ```python
   from agent_gpt import AgentGPT

   estimator = AgentGPT.train_on_cloud(sagemaker_cfg, hyperparams)
   print("Training job:", estimator.latest_training_job.name)

---

## Key Features

### One-Click Training Process
Upload your environment and let AgentGPT handle everything—from setup to job submission—so you can focus on designing better RL tasks.

### Single-Line Code
Launch large-scale distributed training with almost no boilerplate, dramatically reducing the complexity of your workflow.

### Flexible Spaces
Fully supports discrete, continuous, and multiple hybrid action/observation spaces, enabling a wide range of RL tasks.

### Easy Unity ML-Agents Integration
Automatically detects and configures Unity ML-Agents, sparing you from manual environment tinkering.

### Custom Gym Environment Registration
Easily incorporate your custom environment IDs into Gym’s registry. By following Gymnasium’s `register()` conventions, AgentGPT can dynamically launch your environments for distributed training.

### Gymnasium-Compliant
Adheres to Gymnasium v1.0 (and older) standards, ensuring easy integration and interoperability.

### Automatic RL Parameter Tuning
Dynamically adjusts or learns RL hyperparameters—like gamma, lambda, and reward scaling—for optimized performance.

### Robust Multi-Agent
Ensures consistent indexing for agents’ observations, actions, and rewards—even in large-scale, **asynchronous multi-environment** settings.

### Remote Environment Control and Error Handling
Remotely manage environments during training, minimizing downtime and protecting against crashes—even when running on separate machines or cloud instances.

### Advanced GPT API
Manages a GPT-based model—similar to ChatGPT’s query processing—that **dispatches real-time action instructions** from the cloud to your agents. Offers a clean, high-level interface for beginners, alongside advanced hooks for seasoned developers requiring deeper customization.

### Cost Efficiency
Whether you’re a small startup or a large enterprise, you can run hundreds of local environments (for data security or compliance) linked to a single cloud-based trainer. This approach drastically cuts costs, since you won’t pay for hundreds of cloud instances. AWS cost-optimizations and pay-as-you-go pricing further ensure that large-scale RL remains feasible—even on a tight budget.

---

5. **Support**  
For questions, issues, or assistance, you can reach us at:

**Email:** [michikoleo@ccnets.org](mailto:michikoleo@ccnets.org)

Our team typically responds within 1-2 business days. Your feedback is invaluable and helps us continually enhance AgentGPT.
