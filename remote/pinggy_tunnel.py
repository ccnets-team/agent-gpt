# pinggy_tunnel.py

import subprocess
import re
import time

class PinggyTunnel:
    """
    A simple class to spawn a Pinggy tunnel in a Python subprocess.
    It runs the command:
        ssh -p 443 -R0:localhost:<local_port> a.pinggy.io

    Keep in mind:
    - This is a blocking SSH session. If it dies or is closed, the tunnel ends.
    - The output format from Pinggy can change. We try a naive approach to parse a URL.
    """

    def __init__(self, local_port=3000):
        """
        :param local_port: The local port of your app, e.g. 3000 for a React dev server.
        """
        self.local_port = local_port
        self.process = None
        self.public_url = None

    def open_tunnel(self):
        """
        Launch the Pinggy tunnel in a subprocess. We'll read stdout lines
        and look for a public URL pattern. This command typically keeps running
        until you terminate it.
        """
        cmd = [
            "ssh", 
            "-p", "443", 
            f"-R0:localhost:{self.local_port}", 
            "a.pinggy.io"
        ]
        print(f"[PinggyTunnel] Spawning: {' '.join(cmd)}")

        # Spawn the process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Read lines until we find something that looks like a URL, or until process ends
        while True:
            line = self.process.stdout.readline()
            if not line:
                # Process ended or no more output
                break

            # Print the SSH output (for debugging)
            print(f"[PinggyTunnel] {line.strip()}")

            # Example pattern: "Forwarding HTTP traffic from https://abc.pinggy.io"
            # This is just hypothetical -- actual output may differ.
            url_match = re.search(r"(https?://[^\s]+)", line)
            if url_match:
                self.public_url = url_match.group(1)
                print(f"[PinggyTunnel] Public URL captured: {self.public_url}")

        # If we get here, the process might have exited or we reached EOF on stdout
        if self.process.poll() is not None:
            print("[PinggyTunnel] The SSH process ended. No tunnel is active.")

    def close_tunnel(self):
        """
        Terminate the SSH tunnel if it is still running.
        """
        if self.process and self.process.poll() is None:
            print("[PinggyTunnel] Terminating SSH tunnel...")
            self.process.terminate()
            self.process.wait(timeout=5)
            print("[PinggyTunnel] Tunnel closed.")

if __name__ == "__main__":
    # Example usage
    tunnel = PinggyTunnel(local_port=3000)
    tunnel.open_tunnel()

    # The script will stay in open_tunnel() reading lines until the SSH session ends.
    # If you want a quick demo, you might do something like:
    time.sleep(10)  # keep the tunnel open for 10 seconds
    tunnel.close_tunnel()
