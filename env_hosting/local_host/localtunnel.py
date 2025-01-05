# env_hosting/local_host/localtunnel.py
import re
import subprocess
import shutil
import time

class LocalTunnel:
    def __init__(self, host="localhost", port=8000):
        """
        :param host: Where your local server is listening (e.g., localhost, 0.0.0.0)
        :param port: The port your local server is on
        """
        self.host = host
        self.port = port
        self.lt_process = None
        self.assigned_url = None

    def __del__(self):
        self.stop_localtunnel()

    def open_tunnel(self):
        """
        Opens a LocalTunnel process that forwards traffic from a public URL to host:port.
        Returns the discovered localtunnel URL (e.g., https://*****.loca.lt).
        """
        lt_path = shutil.which("lt")
        if lt_path is None:
            raise FileNotFoundError(
                "[LocalTunnelApp] 'lt' not found. "
                "Install LocalTunnel globally (npm install -g localtunnel) and ensure it's on PATH."
            )

        cmd = [
            lt_path,
            "--port", str(self.port),
            "--local-host", self.host,
            "--print-requests"
        ]

        print(f"[LocalTunnelApp] Starting LocalTunnel with command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,   # <--- automatically decode to string
            bufsize=1    # line-buffered for near real-time reading
        )

        self.lt_process = process
        start_time = time.time()

        while True:
            line = process.stdout.readline()
            if not line:
                # Process may have ended or closed stdout
                break

            line_stripped = line.strip()
            if line_stripped:
                print(f"[LocalTunnel OUTPUT] {line_stripped}")

                match = re.search(r"(https:\/\/.*\.loca\.lt)", line_stripped)
                if match:
                    self.assigned_url = match.group(1)
                    print(f"[LocalTunnelApp] Assigned URL: {self.assigned_url}")
                    break

            # Optional: add a timeout to avoid waiting forever
            if (time.time() - start_time) > 30:
                print("[LocalTunnelApp] Timed out waiting for LocalTunnel URL.")
                break

        return self.assigned_url

    def stop_localtunnel(self):
        """
        Gracefully stop the LocalTunnel process.
        """
        if self.lt_process and self.lt_process.poll() is None:
            print(f"[LocalTunnelApp] Stopping LocalTunnel (PID={self.lt_process.pid})...")
            self.lt_process.terminate()
            try:
                self.lt_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[LocalTunnelApp] Force-killing LocalTunnel process.")
                self.lt_process.kill()
            self.lt_process = None
            print("[LocalTunnelApp] LocalTunnel stopped.")

# ------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    tunnel = LocalTunnel(host="localhost", port=8000)
    public_url = tunnel.open_tunnel()
    if public_url:
        print(f"[Main] LocalTunnel URL is: {public_url}")
    else:
        print("[Main] No LocalTunnel URL found (or process timed out).")

    try:
        # Keep the tunnel open until user stops the script or calls stop_localtunnel().
        input("Press ENTER to stop the tunnel...\n")
    finally:
        tunnel.stop_localtunnel()
