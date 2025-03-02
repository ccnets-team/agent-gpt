import logging
logging.getLogger().setLevel(logging.WARNING)

import os
import yaml
import subprocess
import typer
from typing import Optional
from pyngrok import ngrok

def get_stored_ngrok_token() -> Optional[str]:
    """
    Attempts to read the stored ngrok authtoken from common configuration file paths.
    Supports configuration files with either of the following structures:
    
      1. Nested under "agent":
         version: "3"
         agent:
             authtoken: YOUR_TOKEN

      2. Flat structure:
         version: "2"
         region: uus
         authtoken: YOUR_TOKEN

    Returns the token if found; otherwise, returns None.
    """
    potential_paths = []
    if os.name == "nt":  # Windows
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            potential_paths.append(os.path.join(local_appdata, "ngrok", "ngrok.yml"))
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            potential_paths.append(os.path.join(userprofile, ".ngrok2", "ngrok.yml"))
    else:
        # For Linux/macOS
        potential_paths.append(os.path.expanduser("~/.ngrok2/ngrok.yml"))
        potential_paths.append(os.path.expanduser("~/.config/ngrok/ngrok.yml"))
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    config = yaml.safe_load(f)
                    # Try to read the token from nested "agent" key first, then from top-level
                    token = config.get("agent", {}).get("authtoken")
                    if not token:
                        token = config.get("authtoken")
                    if token:
                        return token
            except Exception as e:
                typer.echo(f"Error reading ngrok config file at {path}: {e}")
    return None

def add_ngrok_token(auth_token: str):
    """
    Saves the provided ngrok authtoken permanently using the ngrok CLI.
    """
    try:
        subprocess.run(
            ["ngrok", "config", "add-authtoken", auth_token],
            check=True
        )
    except subprocess.CalledProcessError:
        typer.echo("Failed to add authtoken. Please check your token and try again.")
        raise typer.Exit(code=1)

def get_or_prompt_ngrok_token() -> str:
    """
    Retrieves the stored ngrok token from file. If not found, prompts the user for their token,
    saves it, and returns it.
    """
    token = get_stored_ngrok_token()
    if token:
        return token
    else:
        auth_token = typer.prompt(
            "Enter your ngrok authtoken (visit https://dashboard.ngrok.com/get-started/your-authtoken)",
            hide_input=True
        )
        # Set token for the current session and save it permanently via the CLI
        ngrok.set_auth_token(auth_token)
        add_ngrok_token(auth_token)
        return auth_token
    
def create_tunnel(ports: list) -> str:
    if not ports or len(ports) == 0:
        raise ValueError("At least one port must be provided to create a tunnel.")
    # Retrieve stored token or prompt once
    token = get_or_prompt_ngrok_token()
    # Set the token for the current session (if not already)
    ngrok.set_auth_token(token)
    # Create a tunnel on the first port (reuse its URL for all environments)
    tunnel = ngrok.connect(ports[0])
    return tunnel.public_url
