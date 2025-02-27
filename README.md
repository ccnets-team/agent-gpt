# AgentGPT: Cloud RL Training with Local Env Simulators

**W&B Humanoid-v5 Benchmark (via Internet):** [Weights & Biases Dashboard](https://wandb.ai/junhopark/agentgpt-beta)

![How AgentGPT Works](https://imgur.com/r4hGxqO.png)
---

## Overview

AgentGPT is a one-click, cloud-based platform for distributed reinforcement learning. It lets you easily host your environment simulators—either locally or in the cloud—and connect them to a central training job on AWS SageMaker. This enables efficient data collection and scalable multi-agent training using a GPT-based RL policy.

## Installation

```markdown
pip install agent-gpt-aws --upgrade
```

### Configuration

- **Config hyperparams & SageMaker:**
  ```bash
  agent-gpt config --batch_size 256
  agent-gpt config --role_arn arn:aws:iam::123456789012:role/AgentGPTSageMakerRole
  ```
- **List & Clear current configuration:**
  ```bash
  agent-gpt list
  agent-gpt clear
  ```

### Simulation

- **Run your environment (gym/unity/unreal, etc.) before training starts:**   
  Replace `port1 port2 ...` with actual port values for local environment hosting.
  ```bash
   agent-gpt simulate local port1 port2 ...
  ```

### Training & Inference

- **Set local and cloud environment hosts:**  
  Here, the number at the end indicates the number of agents to run in that environment.
  ```bash
  agent-gpt config --set_env_host local1 http://your_local_ip:port num_agents
  agent-gpt config --set_env_host cloud1 your_endpoint_on_cloud 32
  agent-gpt config --set_env_host cloud2 your_endpoint_on_cloud 64
  ```

- **Train a gpt model on AWS:**
  ```bash
  agent-gpt train
  ```

- **Run agent gpt on AWS:**
  ```bash
  agent-gpt infer
  ```

## Key Features

- **Cloud & Local Hosting:** Quickly deploy environments (Gym/Unity) with a single command.
- **Parallel Training:** Connect multiple simulators to one AWS SageMaker trainer.
- **Real-Time Inference:** Serve a GPT-based RL policy for instant decision-making.
- **Cost-Optimized:** Minimize expenses by centralizing training while keeping simulations local if needed.
- **Scalable GPT Support:** Train Actor (policy) and Critic (value) GPT models together using reverse transitions.
