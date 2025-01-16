# Glossary of Terms 

## A
**AgentGPT** 
A cloud-based system for training and deploying reinforcement learning agents, optimized for efficiency and seamless integration with AWS SageMaker. 

## B
**Batch Size** 
The number of samples processed before the model updates. Used in training reinforcement learning models. 

## C
**Causal Graphs** 
Graphical representation of causal relationship between variables (e.g., states, actions, and rewards) in a system. 

**Causal Reinforcement Learning (CRL)** 
An approach to reinforcement learning that integrates causal inference to improve decision making, sample efficiency, and generalization. 

**CloudEnvLauncher** 
A utility for automating the deployment of environments on the cloud using Docker, Amazon ECR, and EC2. 

**Configurations** 
Predefined settings such as `SageMakerConfig` and `Hyperparameters` that define the parameters for model training, environment hosting, and inference. 

**Counterfactual Reasoning**
The ability to simulate and analyze the outcome of alternative actions (“What if i had taken a different action?”)

## D
**Docker** 
A platform for containerizing applications and their dependencies, ensuring consistent deployment across environments. 

**Domain Knowledge** 
Specific insights or expertise used to design or improve causal models or algorithms. 

## E
**ECR (Elastic Container Registry**
A docker container registry service by AWS that simplifies storing, managing, and deploying container images. 

**EC2 (Elastic Compute Cloud)** 
A web service by AWS that provides resizable compute capacity in the cloud. 

**Episode** 
A sequence of states, actions, and rewards from the start to the termination of an agent’s task in reinforcement learning.

**Environment** 
The system or simulation in which an RL agent operates, such as a game or robot simulation. 

## F
**FastAPI** 
A high-performance web framework for building APIs, used for hosting local environments. 

**Final Observation** 
The last observation of an agent before the environment resets at the end of an episode. 

## G
**Gymnasium API** 
A standardized interface for reinforcement learning environments, with methods like `reset()` and `step()`.

## H
**Hyperparameters** 
Configuration values that control training processes, such as `batch_size`, `max_steps`, and exploration strategies. 

## I
**Info Dictionary** 
A dictionary used in Gymnasium environments to provide additional information, which is optionally returned during `reset()` and `step()`. 

## J
**JSON Serialization** 
The process of converting data into a JSON-compatible format for HTTP communication between systems. 

## L
**Local Environment Hosting**
Hosting the RL environment locally, often with tools like FastAPI and tunnels (e.g., ngrok or localtunnel) for external access. 

## M
**Max Steps** 
The maximum number of steps an RL agent can take in an episode or across training. 

**Missing Observations** 
Instances where an agent has no valid observation data; typically handled using `None`.

## O
**Observation**
Data representing the state of the environment as seen by the RL agent. 

**One-Click Training Process** 
A simplified training workflow requiring minimal configuration and setup. 

## P
**Policy Optimization** 
The process of improving an RL agent’s policy using techniques like policy gradient and Q-learning. 

## R
**Rewards** 
Numerical values indicating the success or failure of an agent’s actions in achieving its goal. 

**Role ARN (Amazon Resource Name)** 
An identifier for an AWS IAM role that grants permissions for specific actions, such as accessing SageMaker resources. 

## S
**SageMaker** 
An AWS service for building, training, and deploying machine learning models at scale. 

**Sample Efficiency** 
The ability to learn effective policies using fewer training samples. 

**SCMs (Structural Causal Models)** 
Mathematical frameworks used to describe causal relationships in reinforcement learning environments. 

**Step Function** 
The method `step(action)` in Gymnasium, which advances the environment by one time step based on an agent’s action. 

## T
**Terminated** 
A flag indicating that the episode has ended due to task completion. 

**Truncated** 
A flag indicating that the episode has ended prematurely, often due to reaching a step limit. 

## U
**Unity ML-Agents** 
A toolkit for developing RL environments in Unity, compatible with Gymnasium API standards. 

## W
**Weight & Biases (W&B)** 
A platform for tracking machine learning experiments and visualizing training metrics in real-time. 
