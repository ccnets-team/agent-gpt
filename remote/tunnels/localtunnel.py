# localtunnel
import re
import subprocess
# conda install -c conda-forge nodejs
# npm install -g localtunnel

class LocalTunnelApp:
    def __init__(self, port=5000):
        self.port = port
        self.lt_process = None
    
    def __del__(self):
        self.stop_localtunnel()

    def start_localtunnel(self):
        import shutil
        lt_path = shutil.which("lt")
        if lt_path is None:
            raise FileNotFoundError(
                "Could not find 'lt' in PATH. Make sure you have localtunnel installed globally "
                "and that it’s on your PATH."
            )

        cmd = [lt_path, "--port", str(self.port)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        url_pattern = re.compile(r'your url is: (https?://[^\s]+)')

        while True:
            line = process.stdout.readline()
            if not line:
                break
            match = url_pattern.search(line)
            if match:
                tunnel_url = match.group(1)
                print(f"[LocalTunnelFlaskApp] LocalTunnel URL: {tunnel_url}")
                self.lt_process = process
                return tunnel_url

        stderr_output = process.stderr.read()
        process.wait()

        # At this point, stderr_output won't crash due to decoding errors
        raise RuntimeError(
            f"Could not parse LocalTunnel URL from output. STDERR:\n{stderr_output}"
        )

    def stop_localtunnel(self):
        if self.lt_process and self.lt_process.poll() is None:
            print(f"[LocalTunnelFlaskApp] Stopping localtunnel (pid={self.lt_process.pid})...")
            self.lt_process.terminate()
            self.lt_process.wait()
            self.lt_process = None
            print("[LocalTunnelFlaskApp] LocalTunnel terminated.")
