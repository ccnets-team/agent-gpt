# AgentGPT: Remote Env Integrated Cloud RL Training

**W&B Humanoid-v5 Benchmark (via Internet):** [Weights & Biases Dashboard](https://wandb.ai/junhopark/agentgpt-beta)

![How AgentGPT Works](https://imgur.com/r4hGxqO.png)
---

## Overview

AgentGPT is a one-click, cloud-based platform for distributed reinforcement learning. It lets you easily host your environment simulators—either locally or in the cloud—and connect them to a central training job on AWS SageMaker. This enables efficient data collection and scalable multi-agent training using a GPT-based RL policy.

## Installation

```markdown
pip install agent-gpt-aws --upgrade
```

### Simulation

- **Launch your environment simulator (e.g., Gym, Unity, Unreal) before training begins:**  
  With this command, your local machine automatically connects to our AgentGPT WebSocket server on the cloud. This real-time connection enables seamless data communication between your environment's state and the cloud training actions, ensuring that everything is ready for the next `agent-gpt train` command.

  ```bash
   agent-gpt simulate
  ```

### Training & Inference

- **Train a gpt model on AWS:**
  ```bash
  agent-gpt train
  ```

- **Run agent gpt on AWS:**
  ```bash
  agent-gpt infer
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

## Key Features

- **Cloud & Local Hosting:** Quickly deploy environments (Gym/Unity) with a single command.
- **Parallel Training:** Connect multiple simulators to one AWS SageMaker trainer.
- **Real-Time Inference:** Serve a GPT-based RL policy for instant decision-making.
- **Cost-Optimized:** Minimize expenses by centralizing training while keeping simulations local if needed.
- **Scalable GPT Support:** Train Actor (policy) and Critic (value) GPT models together using reverse transitions.
