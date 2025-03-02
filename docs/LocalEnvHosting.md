# Local Environment Hosting for Cloud Training: Port Forwarding & Tunneling

If you choose to host your environment locally, your local machine will communicate with the AWS cloud trainer. In this guide, we explain how to expose your local simulation to the cloud using two primary techniques—port forwarding and tunneling—with an emphasis on keeping local hosting as simple as possible.

---

## Overview

In a multi‑agent reinforcement learning workflow, it is commonly assumed that simulations run alongside the training process in a single, self-contained system. Even when leveraging distributed learning via cloud services, the simulation typically remains on a dedicated training machine or runs within a container physically connected to that machine. However, by exposing your local simulation’s IP and ports, the cloud trainer can seamlessly pair with your environment remote. In our system, the configuration serves as an intuitive, real-time interface that displays your simulation status. This ensures that the cloud trainer always connects to your simulator seamlessly, even when it is hosted remotely.

---

## What Is Port Forwarding?

Port forwarding maps a port on your router’s public IP address to a port on your local machine. This allows the cloud trainer to reach your simulation directly using the public IP and port.

**Key Points:**
- **Direct Connection:**  
  If your network permits, you can use your public IP as the simulation endpoint.
- **Port Management:**  
  You configure your router to forward a specific port (e.g., 34560) to your local machine’s IP address.
- **Considerations:**  
  This method is most effective for users with the necessary network permissions, though it may be challenging for individual setups due to varying router configurations and port-forwarding rules.

---

## What Is Tunneling?

Tunneling provides a simpler alternative by creating a public URL that redirects traffic to your local machine. **Ngrok** is the most reliable tool for this purpose.

**Advantages of Ngrok:**
- **Simplicity:**  
  Ngrok automatically generates a public URL, eliminating the need for manual router configuration.
- **Ease-of-Use:**  
  Even if your local machine isn’t set up for port forwarding, ngrok makes your simulation accessible from the cloud.
- **Auto-Configuration:**  
  When you launch a simulation, AgentGPT automatically updates your configuration with the ngrok URL.

---

## How It Works in AgentGPT

When you run the simulation command (e.g., `agent-gpt simulate local` or `agent-gpt simulate simulator_id`), the CLI detects your network settings and configures the appropriate connection method:

- **Local Mode:**  
  - **Direct IP (Port Forwarding):**  
    If you’ve configured port forwarding, your public IP is combined with the simulation port to form the endpoint.
  - **Tunnel (Ngrok):**  
    Alternatively, if tunneling is used, a public URL (e.g., `https://abcd1234.ngrok-free.app:34560`) is generated automatically.

- **Configuration Output:**  
  Your simulator settings are auto-updated in the configuration. For example, running the `agent-gpt list` command might display:

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
  internal_ip: 232.30.12.71
  
  **hyperparams**:
  env_id: Walker2d-v5
  env_entry_point: null
  env_dir: null
  env_hosts:
    local:34560:
      env_endpoint: https://106c-220-119-175-171.ngrok-free.app:34560
      num_agents: 64
    local:34561:
      env_endpoint: https://106c-220-119-175-171.ngrok-free.app:34561
      num_agents: 64
    local:34562:
      env_endpoint: https://106c-220-119-175-171.ngrok-free.app:34562
      num_agents: 64
    local:34563:
      env_endpoint: https://106c-220-119-175-171.ngrok-free.app:34563
      num_agents: 64
  ```

This auto-configuration minimizes manual setup and ensures the cloud trainer can reliably connect to your local simulation.

---

## Practical Steps

### Using Port Forwarding

1. **Access Your Router’s Admin Panel:**  
   Log in to your router’s configuration interface.
2. **Set Up Port Forwarding:**  
   Map a chosen public port (e.g., 34560:34570) to your local machine’s IP address and the port where your simulation is running.
3. **Test the Connection:**  
   Verify that external traffic to your router’s public IP and the chosen port is successfully forwarded to your local simulation.

### Using Ngrok for Tunneling

When you run the agent-gpt simulate command with your "connection" set to "tunnel", the CLI automatically launches ngrok—prompting for a token if required—generates a public URL, opens the server for remote environment communication, and updates the configuration accordingly. This means you don't need to run ngrok manually—the process is fully automated so you can focus on your simulation.

For reference, here are the manual steps (optional) if you ever need to set up a tunnel independently:

1. **Install Ngrok:**  
   Download and install ngrok from [ngrok.com](https://ngrok.com/).
2. **Launch Ngrok:**  
   Run the following command in your terminal (replace `34560` with your simulation port):
   ```bash
   ngrok http 34560
   ```
3. **Obtain the Public URL:**  
   Ngrok will display a public URL (e.g., `https://abcd1234.ngrok-free.app:34560`), which you can verify routes correctly to your local simulation.
4. **Verify Connectivity:**  
   Confirm that the public URL correctly connects to your simulation.

---

## Summary

Local environment hosting for cloud training is made possible through port forwarding and tunneling. With port forwarding, your router maps a public port to your local machine, while tunneling tools like ngrok generate an easy-to-use public URL. AgentGPT automatically updates your simulator registry (as seen via the `agent-gpt list` command) to reflect these settings, ensuring that the AWS cloud trainer can connect to your local simulation. This streamlined process reduces manual configuration and simplifies your workflow, allowing you to focus on training and inference.
