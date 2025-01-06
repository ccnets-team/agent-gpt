import platform
import subprocess

class WindowsFirewallController:
    """
    Manages adding/removing inbound firewall rules on Windows.
    On other OSes, these methods can no-op or raise an exception.
    """
    def __init__(self):
        self.os_name = platform.system().lower()

    def add_inbound_rule(self, port: int, rule_name: str) -> None:
        """
        Creates an inbound rule in Windows Firewall to allow TCP traffic on the given port.
        Requires admin privileges (on Windows). For non-Windows, no-op or raise an error.
        """
        if "windows" not in self.os_name:
            # Either do nothing or raise an error
            print(f"[Info] Not Windows, skipping firewall rule for port {port}.")
            return

        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={rule_name}",
            "dir=in",
            "action=allow",
            "protocol=TCP",
            f"localport={port}"
        ]
        subprocess.run(cmd, check=True)
        print(f"[OK] Firewall rule created: {rule_name} (port {port}, TCP)")

    def remove_inbound_rule(self, port: int, rule_name: str) -> None:
        """
        Removes an inbound rule for the specified port from Windows Firewall.
        Requires admin privileges. For non-Windows, no-op or raise an error.
        """
        if "windows" not in self.os_name:
            # Either do nothing or raise an error
            print(f"[Info] Not Windows, skipping removal of firewall rule for port {port}.")
            return

        cmd = [
            "netsh", "advfirewall", "firewall", "delete", "rule",
            f"name={rule_name}",
            "protocol=TCP",
            f"localport={port}"
        ]
        subprocess.run(cmd, check=True)
        print(f"[OK] Firewall rule removed: {rule_name} (port {port}, TCP)")
