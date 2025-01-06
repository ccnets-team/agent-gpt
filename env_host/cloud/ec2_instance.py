
# env_host/cloud/ec2_instance.py
import boto3
from botocore.exceptions import ClientError
from config.aws_config import EC2Config
from typing import Optional

class EC2Instance:
    """
    Base class for launching a client environment on AWS EC2. 
    Focuses on:
      1) Creating or reusing a security group that allows HTTP (80, 443),
      2) Launching an EC2 instance (no SSH by default),
      3) Returning an HTTP endpoint for remote trainers.
    """

    def __init__(self, ec2_config: EC2Config):
        self.ec2_config = ec2_config
        self.ec2_client = boto3.client("ec2", region_name=ec2_config.region_name)
        self.instance_id: Optional[str] = None

    def launch_ec2_instance(self, ec2_name: str = "my-env-instance", user_data: Optional[str] = None) -> str:
        """
        Launches an EC2 instance using the specified EC2Config, optionally 
        injecting a user_data script to run automatically at startup.
        
        :param ec2_name: Tag or logical name for the instance
        :param user_data: A shell script / cloud-init config to run on instance boot
        :return: The instance ID of the launched EC2 instance.
        """
        # If no SG was provided, create or get one that allows HTTP
        sg_id = self.ec2_config.security_group_id
        if not sg_id:
            sg_id = self._create_or_fetch_security_group()
            self.ec2_config.security_group_id = sg_id

        # Prepare parameters
        params = {
            "ImageId": self.ec2_config.ami_id,
            "InstanceType": self.ec2_config.instance_type,
            "SecurityGroupIds": [sg_id],
            "MinCount": 1,
            "MaxCount": 1,
        }
        if self.ec2_config.key_name:
            params["KeyName"] = self.ec2_config.key_name  # if you want SSH
        if self.ec2_config.subnet_id:
            params["SubnetId"] = self.ec2_config.subnet_id
        if user_data:
            params["UserData"] = user_data

        # Launch
        response = self.ec2_client.run_instances(**params)
        instance_id = response["Instances"][0]["InstanceId"]
        self.instance_id = instance_id
        print(f"[EC2EnvLauncher] Launched EC2 instance: {instance_id}")

        # Name the instance (optional)
        self.ec2_client.create_tags(
            Resources=[instance_id],
            Tags=[{"Key": "Name", "Value": ec2_name}]
        )

        # Wait until it's running
        self._wait_for_running(instance_id)
        return instance_id

    def _create_or_fetch_security_group(self) -> str:
        """
        Creates or reuses a security group that opens inbound ports 80 and 443 for HTTP/HTTPS.
        """
        group_name = "env-hosting-sg"
        try:
            resp = self.ec2_client.create_security_group(
                GroupName=group_name,
                Description="Security group for environment hosting over HTTP",
            )
            sg_id = resp["GroupId"]
            print(f"[EC2EnvLauncher] Created security group '{group_name}' with ID={sg_id}")
            # Authorize inbound HTTP(80)/HTTPS(443)
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 80,
                        "ToPort": 80,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 443,
                        "ToPort": 443,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    },
                ],
            )
            return sg_id
        except ClientError as e:
            if "InvalidGroup.Duplicate" in str(e):
                # Reuse the existing group
                existing = self.ec2_client.describe_security_groups(
                    Filters=[{"Name": "group-name", "Values": [group_name]}]
                )
                sg_id = existing["SecurityGroups"][0]["GroupId"]
                print(f"[EC2EnvLauncher] Reusing security group '{group_name}' with ID={sg_id}")
                return sg_id
            else:
                raise

    def _wait_for_running(self, instance_id: str) -> None:
        """
        Wait until the EC2 instance reaches the 'running' state.
        """
        ec2_resource = boto3.resource("ec2", region_name=self.ec2_config.region_name)
        instance = ec2_resource.Instance(instance_id)
        print(f"[EC2EnvLauncher] Waiting for instance {instance_id} to reach 'running' state...")
        instance.wait_until_running()
        instance.reload()
        print(f"[EC2EnvLauncher] Instance {instance_id} is now in state: {instance.state['Name']}")

    def get_env_endpoint(self) -> Optional[str]:
        """
        Returns the public DNS of the launched instance as an HTTP endpoint,
        e.g., "http://ec2-xx-yy-zz.amazonaws.com".
        """
        if not self.instance_id:
            print("[EC2EnvLauncher] No instance launched yet.")
            return None

        ec2_resource = boto3.resource("ec2", region_name=self.ec2_config.region_name)
        instance = ec2_resource.Instance(self.instance_id)
        public_dns = instance.public_dns_name
        if public_dns:
            endpoint = f"http://{public_dns}"
            print(f"[EC2EnvLauncher] Environment endpoint: {endpoint}")
            return endpoint
        else:
            print("[EC2EnvLauncher] Instance does not have a public DNS name yet.")
            return None

    def generate_user_data_script(self, remote_image_uri: str) -> str:
        """
        Returns a shell script (User Data) that pulls the Docker image on the EC2 instance 
        and runs the container. This script is passed to `launch_ec2_instance(..., user_data=...)`.
        """
        user_data = f"""#!/bin/bash
echo "Pulling Docker image: {remote_image_uri}"
docker pull {remote_image_uri}

echo "Running container from {remote_image_uri}"
docker run -d -p 80:80 {remote_image_uri}
"""
        return user_data