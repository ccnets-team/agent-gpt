import re
import subprocess
import shutil
import time

class LocalTunnel:
    def __init__(self, host, port, tunnel_config=None):
        self.host = host
        self.port = port
        self.lt_process = None

    def __del__(self):
        self.stop_localtunnel()

    def open_tunnel(self):
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

        print(f"[LocalTunnelApp DEBUG] Command to run: {cmd}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=-1
        )

        # Wait a bit so LocalTunnel can start
        time.sleep(3)

        self.lt_process = process
        assigned_url = None

        try:
            # You reset assigned_url = None again here, which is harmless but redundant.
            assigned_url = None  
            
            while True:
                line = process.stdout.readline()
                if not line:
                    # The process might have ended or closed stdout
                    break

                line_stripped = line.strip()

                # If we haven't found the URL yet, look for it
                if not assigned_url:
                    match = re.search(r"(https:\/\/.*\.loca\.lt)", line_stripped)
                    if match:
                        assigned_url = match.group(1)
                        break

        except KeyboardInterrupt:
            print("[LocalTunnelApp] KeyboardInterrupt. Stopping tunnel...")
            process.kill()

        return assigned_url

    def stop_localtunnel(self):
        if self.lt_process and self.lt_process.poll() is None:
            print(f"[LocalTunnelApp] Stopping Localtunnel (pid={self.lt_process.pid})...")
            self.lt_process.terminate()
            self.lt_process.wait(timeout=5)
            self.lt_process = None
            print("[LocalTunnelApp] LocalTunnel terminated.")
