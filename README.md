# README: Integrating Custom Environments with Our Remote Gymnasium System

**Version:** 0.0  
**Date:** 11/20/2024  

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Environment Requirements](#environment-requirements)  
3. [Integration Guidelines](#integration-guidelines)  
   - [3.1 Implementing Gymnasium-Compatible Methods](#31-implementing-gymnasium-compatible-methods)  
   - [3.2 Handling Missing Observations](#32-handling-missing-observations)  
   - [3.3 Handling Final Observations](#33-handling-final-observations)  
   - [3.4 Autoreset Behavior](#34-autoreset-behavior)  
   - [3.5 Multiple Observations at Episode End](#35-multiple-observations-at-episode-end)  
4. [Communication Protocol](#communication-protocol)  
   - [4.1 Data Serialization](#41-data-serialization)  
   - [4.2 API Endpoints](#42-api-endpoints)
   - [4.3 Server Connection Setup](#43-server-connection-setup)
   - [4.4 Server Connection Workflow](#44-server-connection-workflow)
   - [4.5 Example Code for Connection](#45-example-code-for-connection)
   - [4.6 Environment Launch Options](#46-environment-launch-options)
   - [4.7 Error Handling and Debugging](#47-error-handling-and-debugging)
5. [Examples](#examples)
6. [Frequently Asked Questions](#frequently-asked-questions)
7. [Support](#support)

---

## 1. Introduction
Welcome to the integration guide for connecting your custom reinforcement learning environments to our Remote Gymnasium system. This document provides comprehensive instructions and best practices to ensure seamless communication between your environment and our system, enabling efficient training and evaluation in a scalable, cloud-based infrastructure.

The Remote Gymnasium system is a cloud platform that simplifies the process of training and deploying AI agents. By leveraging the power of AWS, it removes the complexities of managing infrastructure, allowing you to focus entirely on building and refining your reinforcement learning environments and algorithms. Whether you're working on state-of-the-art AI models or prototyping new game mechanics, our system provides the tools to streamline and enhance your workflow.

Our platform is designed with flexibility and robustness in mind, enabling seamless integration with a variety of environments. Whetㅇher you're using popular platforms like Unity ML-Agents and Unreal Engine, or building custom game engines, our system adapts to your needs. Additionally, we provide support for Gymnasium version 1.0 and earlier, ensuring compatibility across legacy and modern setups.

### Key Features:
- **One-Click Training Process:** Streamline workflows with a single click. Upload your environment, start training, and get actionable insights instantly.
- **One-Line Coding Simplicity:** Train agents with just a single line of code, reducing setup complexity and time.
- **Versatile Action and Observation Spaces:** Supports discrete, continuous, and hybrid configurations, adapting to diverse RL tasks seamlessly.
- **Seamless Unity and ML Agent Integration :** Automatically detects and integrates Unity ML-Agent data for effortless setup.
- **Automatic Parameter Tuning:** Automatically optimizes hyperparameters like learning rates, exploration rates, and reward thresholds, letting you focus on environment design.
- **Gymnasium-Compatible Design:** Fully adheres to Gymnasium standards (v1.0 and earlier), ensuring easy integration. 
- **Environment Autoreset and Robust Handling:** Automatically resets environments after episodes, ensuring smooth transitions and minimizing downtime.
- **Multi-Agent Support:** Handles multi-agent environments effortlessly, ensuring consistent indexing and smooth coordination of agent-specific observations, actions, and rewards.
- **Flexible API for Advanced Users:** Provides an easy-to-use API for basic operations while offering advanced customization options for experienced developers.
- **Cost-Optimized Cloud Integration:** Minimizes AWS costs while maximizing performance, with flexible pay-per-use pricing models for expense control.

For more detailed informations : 
https://www.linkedin.com/posts/ccnets_1-click-robotics-activity-7231567120537464832-k-o_?utm_source=share&utm_medium=member_desktop

### Performance Highlights
Our algorithm demonstrates state-of-the-art performance, setting new benchmarks in reinforcement learning. Notably:

- **Achieved a 9500+ episode score on Humanoid-v4 Gymnasium**, showcasing advanced capabilities in solving complex RL tasks.
- **Optimized training speeds:** ~10 seconds per episode for 2D games and ~50 seconds per episode for 3D games.
- Integrated **1-Click Training Process**, simplifying setup and accelerating results for developers and researchers alike.

For more details and a demonstration of our GPT-2-powered agent's benchmark score, check out our latest update on LinkedIn: https://www.linkedin.com/posts/ccnets_gpt-2-agent-benchmark-score-with-1-click-activity-7257653827506429952-Ojet?utm_source=share&utm_medium=member_desktop

### How Does It Work:
1. **Prepare Your Environment (local or cloud)** : Design or Customize your game using Unity ML-Agents, or other platform. However, ensure that it adheres to the Gymnasium API Standards (`/reset()` and `/step()`) 
2. **Train the AgentGPT Model** : Start training the AgentGPT model on the cloud with minimal setup. 
3. **Monitor Progress** : Track training performance in real-time 
4. **Deploy Your Agent** : Once trained, download your agent and deploy it into your game. 

---

## 2. Environment Requirements
To integrate your environment with our system, ensure the following:

- **Gymnasium API Compliance:** Implement the standard Gymnasium methods `reset()` and `step(action)`.  
- **Consistent Agent Indexing:** Ensure agent indices remain consistent across observations, actions, and data structures for multi-agent environments.  
- **Handling Missing Data:** Fill positions with `None` in observations, rewards, etc., for agents without data (e.g., inactive agents).  
- **Autoreset Capability:** Implement automatic resets after episode termination, following Gymnasium conventions.  
- **Data Serialization:** Ensure all data exchanged is serializable to JSON format for HTTP communication.  

---

## 3. Integration Guidelines

### 3.1 Implementing Gymnasium-Compatible Methods
Implement the following Gymnasium methods:

#### `reset(seed=None, options=None)`
- **Purpose:** Resets the environment to an initial state.  
- **Returns:**  
  - `observation`: Initial observation(s).  
  - `info`: Dictionary for additional information (can be empty).  

#### `step(action)`
- **Purpose:** Advances the environment by one time step using the provided action.  
- **Returns:**  
  - `observation`: Observation(s) after the action.  
  - `reward`: Reward(s) obtained.  
  - `terminated`: Boolean(s) indicating episode termination.  
  - `truncated`: Boolean(s) indicating if the episode was truncated.  
  - `info`: Dictionary for additional information.

---

### 3.2 Handling Missing Observations
- Use `None` for agents with missing observations.  
- Maintain consistent array sizes across agents.  

### 3.3 Handling Final Observations
At the end of an episode, if your environment provides a final observation:

- **Gymnasium 1.0 Standards:**  
  - The environment should automatically reset after an episode ends.
  - The next `observation` returned will be the initial observation of the new episode.
  - `final_observation` in the `info` dictionary has been **deprecated**.
  - 
- **Pre-Gymnasium 1.0 Standards:**  
  - If the environment does not reset immediately, include a `final_observation` field in the `info` dictionary.
  - This field should contain the last observation(s) before the reset.  

Our system supports both behaviors and handles `final_observation` if provided.

---

### 3.4 Autoreset Behavior
Autoreset ensures that environments are ready for the next step without manual intervention:

- **Automatic Reset:**  
  - Automatically reset the environment after an episode ends (when `terminated` or `truncated` is `True`).
  - This avoids delays and ensures seamless episode transitions.

- **Observation After Reset:**  
  - The observation returned after reset must be the initial observation of the new episode.

- **No Missing Data:**  
  - Ensure that there is no delay or gap in the observation data during the reset process.

---

### 3.5 Multiple Observations at Episode End
In rare cases, environments may produce multiple observations at the end of an episode:

- **Provide the Latest Observation:**  
  - Always return the most recent observation along with the corresponding reward and flags.

- **Consistency:**  
  - Ensure that observation arrays remain consistent in size and agent indexing.

- **Clarify in `info`:**  
  - If the behavior is unusual, include additional details in the `info` dictionary to help interpret the output.

---

## 4. Communication Protocol

### 4.1 Data Serialization
To facilitate seamless communication, all data exchanged between your environment and our system must follow these guidelines:

- **Format:** Use JSON for data serialization.  
- **NumPy Arrays:** Convert NumPy arrays to Python lists before serialization.  
- **Preserve `None`:** Ensure that `None` values are retained to indicate missing or inactive data.  

---

### 4.2 API Endpoints
Our system interacts with your environment using the following HTTP endpoints:

| **Endpoint**       | **Purpose**                                 |  
|---------------------|---------------------------------------------|  
| `/make`            | Initialize a new environment instance.      |  
| `/make_vec`        | Initialize a vectorized environment.        |  
| `/reset`           | Reset the environment to its initial state. |  
| `/step`            | Advance the environment by one time step.   |  
| `/action_space`    | Retrieve the action space specifications.   |  
| `/observation_space` | Retrieve observation space specifications. |  
| `/close`           | Close the environment instance.             |  

---
### 4.3 Server Connection Setup 
**Base URL**

The server communicates through HTTP endpoints. Use the following base URL to interact with the Remote Gymnasium system: 
```
Base URL: https://your-api-domain.com
```
**Authentication**

To access the API, use an API key for authentication. Include the API key in the request headers:

```json
{
    "Authorization": "Bearer YOUR_API_KEY"
}
```
Obtain your API key by contacting the system administrator.

---
### 4.4 Server Connection Workflow
**Initialize Environment** 

Send a `POST` request to `/make` or `/make_vec` to initialize a new environment instance. 

```json
POST https://your-api-domain.com/make
{
    "env_name": "CartPole-v1",
    "seed": 42
}
```

**Reset Environment**

Send a `POST` request to `/reset` to reset the environment to its initial state.

```json
POST https://your-api-domain.com/reset
```

**Take Steps in the Environment**

Send a `POST` request to `/step` with the action to execute.

```json
POST https://your-api-domain.com/step
{
    "action": 0
}
```
**Retrieve Metadata** 

Use `/action_space` and `/observation_space` endpoints to get the specifications of the action and observation spaces.

---
### 4.5 Example Code for Connection 

```python

import requests

BASE_URL = "https://your-api-domain.com"
API_KEY = "YOUR_API_KEY"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Initialize environment
response = requests.post(f"{BASE_URL}/make", headers=headers, json={"env_name": "CartPole-v1"})
print("Environment Initialized:", response.json())

# Reset environment
response = requests.post(f"{BASE_URL}/reset", headers=headers)
print("Reset State:", response.json())

# Step in environment
action = 0
response = requests.post(f"{BASE_URL}/step", headers=headers, json={"action": action})
print("Step Result:", response.json())
```
---
### 4.6 Environment Launch Options 

```python

class EnvironmentLauncher:
   """
   Environment Launcher : Methods to launch the AgentGPT environment in various modes.
   """ 
   @staticmethod
   def launch_on_local_with_ip():
      """
      Host your environment on your local machine, Using the environment IP address as the endpoint (Public IP and Port). The firewall and port will be temporartily opened to allow communication during the operation, and securely closed afterward to maintain network security. Note : Ngrok is used for secure tunneling in this mode.
      """
      pass

   @staticmethod
   def launch_on_local_with_url():
      """
      Runs the algorithm locally while exposing it to external systems. The environment endpoint is a temporary public URL (e.g., 'https://'), which allows external systems to communicate with the local instance.
      """
      pass

   @staticmethod
   def launch_on_cloud():
      """
      Deploys the algorithm in an AWS environment, providing a globally accessible endpoint via a public URL or API Gateway. This setup ensures scalability, high availability, and seamless integration with AWS services like S3, ECR, and EC2
      """
      pass
```
---
### 4.7 Error Handling and Debugging 

**Common Errors** 
- **401 Unauthorized:**  Ensure your API key is valid and included in the `/Authorization` header.
- **404 Not Found:** Verify the endpoint path and the base URL.
- **500 Internal Server Error:** Check server logs or contact support.

**Debugging** 
- Enable verbose logging for HTTP requests (e.g., `/requests` debug logs in Python).
- Use tools like Postman or cURL to test API endpoints manually.

## 5. Documentations

---
## 6. Frequently Asked Questions

**Q:** What should I do if an agent has no action?  
**A:** Use `None` or a default action for such agents. Ensure your environment can process these values appropriately. This allows for consistency in action arrays and smooth handling by our system.

---

**Q:** Can I include additional information in the `info` dictionary?  
**A:** Yes, you can include auxiliary data in the `info` dictionary, as long as all fields are JSON-serializable. This is helpful for providing additional context or debugging information during integration.

---

**Q:** How does the system handle environments not conforming to Gymnasium 1.0?  
**A:** Our system is compatible with both Gymnasium 1.0 and earlier versions. If your environment does not follow Gymnasium 1.0 conventions, include any required fields (e.g., `final_observation`) in the `info` dictionary, and our system will process them accordingly.

---

**Q:** What if my environment produces multiple observations at the end of an episode?  
**A:** Return the most recent observation and ensure consistency with the rewards and termination flags. If the behavior is unusual, provide clarifying details in the `info` dictionary to make the output easier to interpret.

---
## 7. Support 
If you encounter any issues, have questions, or need further assistance, feel free to reach out through :

Email: michikoleo@ccnets.org 

We value your feedback and strive to provide timely responses. Our team typically replies within 1-2 business days. Your inquiries help us improve, so don’t hesitate to reach out!





