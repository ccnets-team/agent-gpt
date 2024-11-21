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
   - [4.6 Error Handling and Debugging](#46-error-handling-and-debugging)
5. [Examples](#examples)
6. [Frequently Asked Questions](#frequently-asked-questions)
7. [Support](#support)

---

## 1. Introduction
Welcome to the integration guide for connecting your custom reinforcement learning environments to our Remote Gymnasium system. This document provides comprehensive instructions and best practices to ensure seamless communication between your environment and our system, facilitating efficient training and evaluation over AWS.

Our system is designed to be flexible and robust, accommodating environments built with various platforms such as Unity ML-Agents, Unreal Engine, or custom game engines. We aim to support both Gymnasium version 1.0 and earlier versions, ensuring broad compatibility.

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
***Header Example***
```json
{
    "Authorization": "Bearer YOUR_API_KEY"
}
Obtain your API key by contacting the system administrator.
```
---
### 4.4 Server Connection Workflow
**Initialize Environment** 
Send a POST request to /make or /make_vec to initialize a new environment instance. 
```json
POST https://your-api-domain.com/make
{
    "env_name": "CartPole-v1",
    "seed": 42
}
```
**Reset Environment**
Send a POST request to /reset to reset the environment to its initial state.
```json
POST https://your-api-domain.com/reset
```
**Take Steps in the Environment**
Send a POST request to /step with the action to execute.
```json
POST https://your-api-domain.com/step
{
    "action": 0
}
```
**Retrieve Metadata** 
Use /action_space and /observation_space endpoints to get the specifications of the action and observation spaces.

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
### 4.6 Error Handling and Debugging 
**Common Errors** 
**401 Unauthorized:**  Ensure your API key is valid and included in the /Authorization header.
**404 Not Found:** Verify the endpoint path and the base URL.
**500 Internal Server Error:** Check server logs or contact support.
**Debugging** 
- Enable verbose logging for HTTP requests (e.g., /requests debug logs in Python).
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





