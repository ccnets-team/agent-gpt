# AgentGPT: Distributed RL training with AWS and local simulators

**Overall Gym Benchmark Archive(running on standard Gym, not via Internet):** [Weights & Biases Dashboard](https://wandb.ai/ccnets/one-click-robotics)

**Live Gym Humanoid-v5(via Internet):** [AWS CloudWatch Dashboard](https://cloudwatch.amazonaws.com/dashboard.html?dashboard=AgentGPT-Benchmark-Gym-Humanoid-v5&context=eyJSIjoidXMtZWFzdC0xIiwiRCI6ImN3LWRiLTUzMzI2NzMxNjcwMyIsIlUiOiJ1cy1lYXN0LTFfcUFYZHp4ank3IiwiQyI6Ijc2bXM5azI2dHE2a29pY2IwZGxkc2g2bDgwIiwiSSI6InVzLWVhc3QtMTo1YTJjZTUxMy04YTE2LTQ1NTEtYWEyNS05Mjk3ZjE3ZjVkNzUiLCJNIjoiUHVibGljIn0%3D)


![How AgentGPT Works](https://imgur.com/r4hGxqO.png)
---

## Overview

AgentGPT is a one-click, cloud-based platform for distributed reinforcement learning. It lets you easily host your environment simulators—either locally or in the cloud—and connect them to a central training job on AWS SageMaker. This enables efficient data collection and scalable multi-agent training using a GPT-based RL policy.

## Key Features

- **Cloud & Local Hosting:** Quickly deploy environments (Gym/Unity) with a single command.
- **Parallel Training:** Connect multiple simulators to one AWS SageMaker trainer.
- **Real-Time Inference:** Serve a GPT-based RL policy for instant decision-making.
- **Cost-Optimized:** Minimize expenses by centralizing training while keeping simulations local if needed.
- **Scalable GPT Decision Model Support:** Use a reverse-environment simulating GPT that trains Actor (policy) and Critic (value) GPT models together, enabling them to learn optimal action sequences through reverse transitions.

## Architecture

1. **Environment Hosting:**
   - **Local:** FastAPI servers (optionally tunneled via ngrok/localtunnel).
   - **Cloud:** Docker containers on AWS (ECR/EC2).

2. **AgentGPT:**
   - Manages AWS SageMaker training jobs and real-time inference endpoints.

3. **GPTAPI:**
   - Provides methods for multi-agent actions like `select_action` and `sample_action`.

4. **Configuration:**
   - **SageMakerConfig:** AWS roles, Docker image URIs, and model artifact paths.
   - **Hyperparameters:** Settings for environment IDs, batch sizes, exploration strategies, etc.

This streamlined platform enables you to leverage cloud power for RL training while using locally hosted simulators for efficient, scalable, and cost-effective multi-agent training.
