import os
import json
import platform
import subprocess

try:
    import miniupnpc
except ImportError:
    miniupnpc = None  # We'll check later if it's installed

OPENED_PORTS_FILE = "opened_ports.json"


class PortForwardController:
    """
    Manages UPnP (universal) port-forwarding logic and
    tracks opened ports in a JSON file, regardless of OS.

    It can optionally invoke a FirewallManager (e.g., WindowsFirewallManager)
    if you pass one in, or if you detect Windows.
    """

    def __init__(self,
                 public_ip=None,
                 internal_ip=None,
                 free_port=None,
                 opened_ports_file=OPENED_PORTS_FILE,
                 firewall_manager=None):
        """
        :param public_ip: (Optional) The WAN or public IP, if known.
        :param internal_ip: (Optional) The LAN or local IP, if needed for UPnP.
        :param free_port: (Optional) A default port number you'd like to open.
        :param opened_ports_file: The JSON file to store opened ports data.
        :param firewall_manager: An optional manager for firewall rules 
               (e.g., WindowsFirewallManager instance).
        """
        self.public_ip = public_ip
        self.internal_ip = internal_ip
        self.free_port = free_port
        self.opened_ports_file = opened_ports_file
        self.firewall_manager = firewall_manager

        self.os_name = platform.system().lower()
        print(f"[PortForwardManager] Detected OS: {self.os_name}")

    # -------------------------------------------------------------------------
    # JSON Persistence
    # -------------------------------------------------------------------------
    def load_opened_ports(self) -> dict:
        """
        Load the JSON file of opened ports.
        Returns a dict with 'firewall_rules' and 'upnp_forwards' lists.
        """
        if not os.path.exists(self.opened_ports_file):
            return {
                "firewall_rules": [],  # e.g. [{port: 8080, name: "OpenPort_8080"}]
                "upnp_forwards": []    # e.g. [{port: 8080}]
            }

        with open(self.opened_ports_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_opened_ports(self, data: dict) -> None:
        """
        Save the dictionary containing opened ports info to JSON.
        """
        with open(self.opened_ports_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------------------
    # UPnP Forwarding
    # -------------------------------------------------------------------------
    def add_upnp_forward(self, port: int, description: str = "MyService") -> bool:
        """
        Attempts to add a TCP port mapping via UPnP.
        Returns True if successful, False otherwise.
        Works on any OS, provided miniupnpc is installed and a UPnP device is found.
        """
        if not miniupnpc:
            print("[Error] miniupnpc not installed. Skipping UPnP mapping.")
            return False

        upnp = miniupnpc.UPnP()
        upnp.discoverdelay = 200
        found = upnp.discover()
        if found == 0:
            print("[Error] No UPnP devices discovered. Skipping UPnP mapping.")
            return False

        upnp.selectigd()
        lan_ip = upnp.lanaddr
        external_ip = upnp.externalipaddress()

        # externalPort=port, protocol='TCP', internalPort=port,
        # internalClient=lan_ip, desc=description
        result = upnp.addportmapping(port, 'TCP', port, lan_ip, description, '')
        if result:
            print(f"[OK] UPnP port {port} forwarded: {external_ip}:{port} -> {lan_ip}:{port}")
            return True
        else:
            print("[Error] Failed to create UPnP mapping.")
            return False

    def remove_upnp_forward(self, port: int) -> bool:
        """
        Removes the UPnP port mapping for the given port (TCP).
        Returns True if successful, False otherwise.
        """
        if not miniupnpc:
            print("[Error] miniupnpc not installed. Can't remove UPnP mapping.")
            return False

        upnp = miniupnpc.UPnP()
        upnp.discoverdelay = 200
        found = upnp.discover()
        if found == 0:
            print("[Error] No UPnP devices discovered. Can't remove UPnP mapping.")
            return False

        upnp.selectigd()
        result = upnp.deleteportmapping(port, 'TCP')
        if result:
            print(f"[OK] Removed UPnP mapping for port {port}")
            return True
        else:
            print(f"[Error] Could not remove UPnP mapping for port {port}")
            return False

    # -------------------------------------------------------------------------
    # Higher-Level Methods: add/remove a port and record in JSON
    # -------------------------------------------------------------------------
    def add_opened_port(self, port: int, name: str = None) -> None:
        """
        Creates a (Windows) firewall rule + UPnP forward for the given port,
        and saves the info to `opened_ports_file`.

        :param port: The port number (int).
        :param name: A name for the rule (str). Defaults to "OpenPort_<port>".
        """
        if name is None:
            name = f"OpenPort_{port}"

        data = self.load_opened_ports()

        # 1) Add firewall rule (if we have a WindowsFirewallManager, or if OS is Windows)
        if self.firewall_manager is not None:
            try:
                self.firewall_manager.add_inbound_rule(port, name)
                data["firewall_rules"].append({"port": port, "name": name})
            except subprocess.CalledProcessError:
                print(f"[Error] Could not add firewall rule for port {port}.")

        # 2) Attempt UPnP forward (optional, any OS)
        if self.add_upnp_forward(port, description=name):
            data["upnp_forwards"].append({"port": port})

        # 3) Save updated data
        self.save_opened_ports(data)
        print(f"[PortForwardManager] Port {port} added (name={name}).")

    def remove_all_opened_ports(self) -> None:
        """
        Reads the opened_ports_file and removes all recorded ports from Windows Firewall + UPnP.
        Then deletes the file.
        """
        if not os.path.exists(self.opened_ports_file):
            print("[Info] No opened_ports.json found. Nothing to remove.")
            return

        data = self.load_opened_ports()

        # Remove firewall rules (if firewall_manager is available)
        if self.firewall_manager is not None:
            for rule in data["firewall_rules"]:
                port = rule["port"]
                name = rule["name"]
                try:
                    self.firewall_manager.remove_inbound_rule(port, name)
                except subprocess.CalledProcessError:
                    print(f"[Error] Could not remove firewall rule (port={port}, name={name}).")

        # Remove UPnP mappings
        for forward in data["upnp_forwards"]:
            self.remove_upnp_forward(forward["port"])

        # Delete the opened ports file
        os.remove(self.opened_ports_file)
        print(f"[OK] All ports removed and {self.opened_ports_file} deleted.")
