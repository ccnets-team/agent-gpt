# AgentGPT: One-Click Cloud-Based Distributed RL

**Repository:** [ccnets-team/agent-gpt](https://github.com/ccnets-team/agent-gpt.git)  

**Overall Gym Benchmark Archive(running on standard Gym, not via Internet):** [Weights & Biases Dashboard](https://wandb.ai/ccnets/one-click-robotics)

**Live Gym Humanoid-v5(via Internet):** [AWS CloudWatch Dashboard](https://cloudwatch.amazonaws.com/dashboard.html?dashboard=AgentGPT-Benchmark-Gym-Humanoid-v5&context=eyJSIjoidXMtZWFzdC0xIiwiRCI6ImN3LWRiLTUzMzI2NzMxNjcwMyIsIlUiOiJ1cy1lYXN0LTFfcUFYZHp4ank3IiwiQyI6Ijc2bXM5azI2dHE2a29pY2IwZGxkc2g2bDgwIiwiSSI6InVzLWVhc3QtMTo1YTJjZTUxMy04YTE2LTQ1NTEtYWEyNS05Mjk3ZjE3ZjVkNzUiLCJNIjoiUHVibGljIn0%3D)

This README explains how AgentGPT orchestrates RL training on AWS SageMaker, hosts environments in the cloud or locally, and provides multi-agent GPT-based endpoints.

![How AgentGPT Works](https://imgur.com/r4hGxqO.png)
---

## Overview
AgentGPT is a **one-click, cloud-based distributed RL** platform built for multi-agent environments running in **parallel**. With minimal setup, it can automatically package and host your environment (assigning it a URL or IP) or launch a training job on AWS SageMaker. A **GPT model** is then served in real time, supporting **multiple local or cloud simulators** connected to a single AgentGPT trainer in the cloud—scaling effortlessly from small proofs-of-concept to large-scale, asynchronous RL deployments.

> **Note for Large Enterprises**  
> You can **keep your actual environment simulations on local machines** (on-premise, behind the firewall, etc.) to maintain security and control data flow. Rather than running hundreds of cloud-hosted environments, you send environment states from your local servers to a **single lightweight trainer** in AWS. This architecture **drastically reduces hosting costs** and provides peace of mind for sensitive data.

**Key Features**:
- **Cloud & Local Environment Hosting** – Deploy Gym/Unity environments with a single command. 
- **Parallel Training** - Multiple environments (local/cloud) connect to a single SageMaker trainer.
- **Real-Time Inference**: GPT-based RL policy served via AWS SageMaker.
- **Cost-Optimized Cloud Usage** - Use one trainer for multiple agents, minimizing expenses. 
- **Multi-Agent & Asynchronous RL Support** - Train hundreds of agents simultaneously. 
- **Distributed RL Agent Support**: Each environment endpoint feeds observations to—and receives actions from—a central policy, enabling fully distributed, scalable training.  

---

## Architecture

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

## Highlights

### One-Click Training Process
Upload your environment and let **AgentGPT** handle everything—**setup, job submission, and training**—so you can focus on designing better RL tasks

### Single-Line Code
Train large-scale, distributed RL models with **minimal boilerplate**, reducing complexity and setup time. 

### Flexible Spaces
Fully supports **discrete, continuous, and hybrid action/observation spaces**, making it adaptable for any RL task. 

### Easy Unity ML-Agents Integration
**Auto-detects and configures Unity ML-Agents**, eliminating manual setup and simplifying game AI training. 

### Custom Gym Environment Registration
Easily register custom environments using **Gymnasium’s `register()` API**, ensuring **fast, scalable training**

### Gymnasium-Compliant
Adheres to Gymnasium v1.0 (and older) standards, ensuring easy integration and interoperability.

### Automatic RL Parameter Tuning
Automatically optimizes hyperparameters like **gamma, lambda, and reward scaling** for peak performance. 

### Robust Multi-Agent
Ensures **consistent indexing** for multiple agents, even in **asynchronous, multi-environment setups**. 

### Remote Environment Control and Error Handling
Manage and control training environments **remotely**, preventing crashes and minimizing downtime. 

### Advanced GPT API
Leverages a **GPT-based model** for real-time **action decision-making**, offering **high-level simplicity** for beginners and **deep customization** for advanced users. 

### Cost Efficiency
Reduce cloud expenses by **linking local environments to a single cloud trainer**, cutting AWS costs **without sacrificing performance**.

---
