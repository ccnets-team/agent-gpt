# remote/localtunnel.py
"""
LocalTunnelApp
--------------
A utility class to open a Node-based localtunnel on the specified port.

**Dependencies**:
    1. Node.js must be installed and on PATH.
    2. npm must also be on PATH (usually included with Node.js).
       - If npm is missing but a conda environment is detected, 
         this script will attempt 'conda install -c conda-forge nodejs'.
       - Otherwise, you must install npm manually.
    3. localtunnel (npm package) can be installed globally 
       (npm install -g localtunnel) or locally in your project:
         npm init -y
         npm install localtunnel

**Usage**:
    from remote.localtunnel import LocalTunnelApp

    app = LocalTunnelApp(port=5000)
    public_url = app.open_localtunnel()
    print("Tunnel URL:", public_url)

    # Later, you can close it:
    app.close()
"""

import os
import re
import subprocess
import shutil
import sys
import time

class LocalTunnelApp:
    """
    A utility class that spawns a Node-based localtunnel via 'npx localtunnel'.

    :param port: The local port to tunnel (e.g., 5000).
    :param project_dir: (Optional) Project folder path for future usage 
                        (not strictly required in this version).
    """

    def __init__(self, port, project_dir=None):
        self.port = port
        self.process = None
        # Default to one level up from this file if not specified
        self.project_dir = project_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.public_url = None

    def open_localtunnel(self):
        """
        Creates a tunnel via 'npx localtunnel'.
        Returns a public URL string, or None on failure.
        """
        node_executable = shutil.which("node")
        npm_executable = shutil.which("npm")

        # 1) Check for Node.js
        if not node_executable:
            print("[LocalTunnelApp] Node.js not found in PATH. Please install Node.js.")
            return None

        # 2) Check for npm
        if not npm_executable:
            print("[LocalTunnelApp] Node.js found, but 'npm' not on PATH.")
            # Attempt conda-based install if conda is detected
            if self._is_conda_environment():
                print("[LocalTunnelApp] Trying to install npm via conda-forge (this also installs nodejs).")
                install_success = self._attempt_conda_install_nodejs()
                if not install_success:
                    print("[LocalTunnelApp] Automatic npm installation failed. "
                          "Please install Node.js + npm manually.")
                    return None
                # Re-check for npm
                npm_executable = shutil.which("npm")
                if not npm_executable:
                    print("[LocalTunnelApp] 'npm' still not found after conda install. "
                          "Please install manually.")
                    return None
            else:
                print("[LocalTunnelApp] Not in a conda environment, cannot auto-install npm. "
                      "Please install npm manually.")
                return None

        print("[LocalTunnelApp] Node.js & npm detected.")
        self.public_url = self._launch_localtunnel_node()
        return self.public_url

    def close(self):
        """
        Closes the localtunnel subprocess if it is still running.
        """
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
            print("[LocalTunnelApp] localtunnel process terminated.")

    def _launch_localtunnel_node(self):
        """
        Spawns localtunnel with a unique, time-based subdomain via 'npx localtunnel'.
        """
        npx_executable = shutil.which("npx")
        if not npx_executable:
            print("[LocalTunnelApp] Node.js & npm installed, but 'npx' not found on PATH.")
            return None

        # Create a unique subdomain string based on current time (microseconds)
        subdomain_base = "agentgpt-ccnets"
        micro_timestamp = int(time.time() * 1_000_000)
        unique_subdomain = f"{micro_timestamp}-{subdomain_base}"

        cmd = [
            npx_executable,
            "localtunnel",
            "--port", str(self.port),
            # "--subdomain", unique_subdomain
        ]

        print(f"[LocalTunnelApp] Spawning localtunnel via npx on port {self.port} ...")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        public_url = None
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            if "url" in line.lower():
                match = re.search(r"(https?://[^\s]+)", line)
                if match:
                    public_url = match.group(1)
                    break

        if not public_url:
            error_output = self.process.stderr.read()
            print("[LocalTunnelApp] Failed to detect public URL from localtunnel output.")
            print("[stderr]:", error_output)
            return None

        print(f"[LocalTunnelApp] localtunnel public URL: {public_url}")
        return public_url

    def _is_conda_environment(self):
        """
        Checks if we're running inside a conda environment.
        Returns True if so, False otherwise.
        """
        return ("CONDA_PREFIX" in os.environ) or ("conda" in sys.executable.lower())

    def _attempt_conda_install_nodejs(self):
        """
        Attempts to install nodejs (which includes npm) using conda-forge.
        Returns True if successful, False otherwise.
        """
        try:
            cmd = ["conda", "install", "-c", "conda-forge", "nodejs", "-y"]
            print(f"[LocalTunnelApp] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return True
        except FileNotFoundError:
            print("[LocalTunnelApp] 'conda' command not found. Not in a conda environment?")
            return False
        except subprocess.CalledProcessError as e:
            print("[LocalTunnelApp] Conda install failed:", e)
            return False
