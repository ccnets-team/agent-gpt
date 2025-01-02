import os
import time
import boto3
import subprocess
from botocore.exceptions import ClientError
from env_hosting.config.ec2_config import EC2Config

class AWSEC2Tunnel:
    """
    Manages an EC2 instance for environment hosting or tunneling.
    Assumes AWS credentials are configured locally (i.e., via 'aws configure').
    """
    def __init__(self, 
                 port, host,
                 ec2_config: EC2Config, instance_tag="AgentGPTHost"):
        """
        :param ec2_config:    A dataclass object containing region, key_name, etc.
        :param instance_tag:  Tag Name=... to identify/create the instance.
        """
        self.local_port = port
        self.remote_port = port
        self.ec2_config = ec2_config
        self.instance_tag = instance_tag

        self.region = ec2_config.region_name
        self.key_name = ec2_config.key_name
        self.instance_type = ec2_config.instance_type
        self.ami_id = ec2_config.ami_id

        # Create resource + client with the specified region
        self.ec2_resource = boto3.resource("ec2", region_name=self.region)
        self.ec2_client = boto3.client("ec2", region_name=self.region)

        self.instance_id = None
        self.public_dns = None
        self.ssh_process = None  # handle to the background SSH process if we open a tunnel

    def create_or_find_ec2_instance(self) -> str:
        """
        Find a running EC2 instance with Tag:Name=<self.instance_tag>.
        If not found, launch a new one.
        :return: The instance_id of the found or newly created instance.
        """
        filters = [
            {"Name": "instance-state-name", "Values": ["running"]},
            {"Name": "tag:Name", "Values": [self.instance_tag]}
        ]
        instances = list(self.ec2_resource.instances.filter(Filters=filters))

        if instances:
            self.instance_id = instances[0].id
            print(f"[INFO] Found existing running instance {self.instance_id} with tag={self.instance_tag}")
            return self.instance_id

        # No instance found; create a new one
        print(f"[INFO] No running instance found with tag={self.instance_tag}, launching a new one...")
        try:
            new_instance = self.ec2_resource.create_instances(
                ImageId=self.ami_id,
                InstanceType=self.instance_type,
                KeyName=self.key_name,  # must exist in AWS; user must have corresponding .pem
                MinCount=1,
                MaxCount=1,
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": self.instance_tag}]
                    }
                ]
            )[0]
            self.instance_id = new_instance.id
            print(f"[INFO] Created new instance: {self.instance_id}. Waiting for it to run...")
        except ClientError as e:
            raise RuntimeError(f"Error creating EC2 instance: {e}")

        # Wait until instance is in "running" state
        new_instance.wait_until_running()
        print(f"[INFO] Instance {self.instance_id} is now running.")
        return self.instance_id

    def wait_for_public_dns(self, timeout_secs=120, interval_secs=5) -> str:
        """
        Wait until the instance has a public DNS name assigned.
        Store in self.public_dns and return it.

        :param timeout_secs:  Max time to wait before raising a RuntimeError.
        :param interval_secs: Interval in seconds between checks.
        :return:             The public DNS of the instance.
        """
        if not self.instance_id:
            raise ValueError("No instance_id available. Did you call create_or_find_ec2_instance()?")

        print(f"[INFO] Checking for public DNS on {self.instance_id}...")

        total_time = 0
        while total_time < timeout_secs:
            resp = self.ec2_client.describe_instances(InstanceIds=[self.instance_id])
            reservations = resp.get("Reservations", [])
            if reservations and "Instances" in reservations[0]:
                instance_data = reservations[0]["Instances"][0]
                dns = instance_data.get("PublicDnsName")
                if dns:
                    self.public_dns = dns
                    print(f"[INFO] Found public DNS: {dns}")
                    return dns
            time.sleep(interval_secs)
            total_time += interval_secs

        raise RuntimeError(
            f"Timeout ({timeout_secs}s): Instance {self.instance_id} does not have a public DNS yet."
        )

    def open_ec2_ssh(
        self,
        ec2_user="ec2-user",
        default_key_path="~/.ssh/my-ec2-key.pem",
    ) -> str:
        local_port = self.local_port
        remote_port = self.remote_port
        """
        Opens a reverse SSH tunnel from the EC2 instance to your local machine.

        The instance's public DNS:remote_port -> localhost:local_port

        :param local_port:  The port on your local machine to forward traffic to.
        :param remote_port: The port on the EC2 instance to listen on.
        :param ec2_user:    The SSH username (typically 'ec2-user' on Amazon Linux).
        :param default_key_path: Default path to your SSH key if not set via EC2_KEY_PATH env var.
        :return:            The public URL (http://<public_dns>:remote_port).
        """
        if not self.instance_id:
            raise ValueError("No instance_id available. Did you call create_or_find_ec2_instance()?")

        if not self.public_dns:
            raise ValueError("No public_dns found. Did you call wait_for_public_dns()?")

        # Determine private key path
        key_path = os.environ.get("EC2_KEY_PATH", "").strip() or os.path.expanduser(default_key_path)
        if not os.path.isfile(key_path):
            raise ValueError(
                "No valid SSH key found. Please specify a valid .pem path by either:\n"
                " - Setting the EC2_KEY_PATH environment variable, or\n"
                f" - Placing your private key at {default_key_path}\n\n"
                f"Current key path tried: {key_path}"
            )

        # Construct SSH command: reverse tunnel
        ssh_command = [
            "ssh",
            "-i", key_path,
            "-o", "StrictHostKeyChecking=no",
            "-R", f"0.0.0.0:{remote_port}:localhost:{local_port}",
            f"{ec2_user}@{self.public_dns}"
        ]

        print("[INFO] Launching reverse SSH tunnel:")
        print(" ".join(ssh_command))

        # Launch SSH in the background
        self.ssh_process = subprocess.Popen(ssh_command)

        # Return public URL
        public_url = f"http://{self.public_dns}:{remote_port}"
        print(f"[INFO] Reverse SSH tunnel established! Public URL: {public_url}")
        return public_url

    def terminate_ssh(self):
        """
        If the SSH tunnel is active, terminate it.
        """
        if self.ssh_process and self.ssh_process.poll() is None:
            print(f"[INFO] Terminating SSH process (pid={self.ssh_process.pid})...")
            self.ssh_process.terminate()
            self.ssh_process.wait()
            self.ssh_process = None
            print("[INFO] SSH tunnel process terminated.")

    def open_tunnel(
        self, do_reverse_tunnel=True
    ) -> str:
        local_port= self.local_port
        remote_port = self.remote_port
        """
        Orchestrates the entire sequence:
         1) Create or find an EC2 instance.
         2) Wait for Public DNS.
         3) Optionally open a reverse SSH tunnel (if do_reverse_tunnel=True).

        :param local_port:       Local port to forward traffic to.
        :param remote_port:      Remote port on the EC2 instance.
        :param do_reverse_tunnel:If True, opens a reverse SSH tunnel.
        :return:                 The public URL for your environment, or direct DNS if no tunnel.
        """
        self.create_or_find_ec2_instance()
        self.wait_for_public_dns()

        if do_reverse_tunnel:
            public_url = self.open_ec2_ssh(
                local_port=local_port,
                remote_port=remote_port
            )
        else:
            # If you're hosting your environment directly on EC2, you might do:
            public_url = f"http://{self.public_dns}:{remote_port}"
            print(f"[INFO] Environment is (or will be) hosted on EC2 at: {public_url}")

        return public_url


def main():
    """
    Example usage:
     - If you want to run environment locally + reverse SSH tunnel, set do_reverse_tunnel=True.
     - If you want to host environment on EC2, set do_reverse_tunnel=False (and run environment on the EC2).
    """

    ec2_config = EC2Config(
        region_name="us-east-1",
        ami_id="ami-08c40ec9ead489470",
        instance_type="t2.micro",
        key_name="MyKeyPair",
    )

    manager = AWSEC2Tunnel(port=5000, host="127.0.0.1", ec2_config=ec2_config, instance_tag="AgentGPTHost")
    
    # Example 1: Reverse SSH tunnel (local environment)
    local_url = manager.open_tunnel(do_reverse_tunnel=True)
    print(f"[MAIN] Your local environment is now accessible at: {local_url}")

    # Example 2: If you want to run environment directly on EC2, skip the tunnel:
    # ec2_direct_url = manager.open_tunnel(local_port=5000, remote_port=5000, do_reverse_tunnel=False)
    # print(f"[MAIN] Your EC2 environment is at: {ec2_direct_url}")

if __name__ == "__main__":
    main()
