
# env_host/cloud/ec2_instance.py
import boto3
from botocore.exceptions import ClientError
from config.aws_config import EC2Config
from typing import Optional

def _create_or_fetch_security_group(ec2_client) -> str:
    """
    Creates or reuses a security group that opens inbound ports 80 and 443 for HTTP/HTTPS.
    """
    group_name = "env-hosting-sg"
    try:
        resp = ec2_client.create_security_group(
            GroupName=group_name,
            Description="Security group for environment hosting over HTTP"
        )
        sg_id = resp["GroupId"]
        print(f"[EC2EnvLauncher] Created security group '{group_name}' with ID={sg_id}")
        # Authorize inbound HTTP(80)/HTTPS(443)
        ec2_client.authorize_security_group_ingress(
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
            existing = ec2_client.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [group_name]}]
            )
            sg_id = existing["SecurityGroups"][0]["GroupId"]
            print(f"[EC2EnvLauncher] Reusing security group '{group_name}' with ID={sg_id}")
            return sg_id
        else:
            raise

def _wait_for_running(instance_id: str, region_name: str) -> None:
    """
    Wait until the EC2 instance reaches the 'running' state.
    """
    ec2_resource = boto3.resource("ec2", region_name=region_name)
    instance = ec2_resource.Instance(instance_id)
    print(f"[EC2EnvLauncher] Waiting for instance {instance_id} to reach 'running' state...")
    instance.wait_until_running()
    instance.reload()
    print(f"[EC2EnvLauncher] Instance {instance_id} is now in state: {instance.state['Name']}")


def launch_ec2_instance_impl(ec2_client, ec2_config: EC2Config, user_data: Optional[str] = None) -> str:
    """
    Launches an EC2 instance using the specified EC2Config, optionally 
    injecting a user_data script to run automatically at startup.
    
    :param ec2_name: Tag or logical name for the instance
    :param user_data: A shell script / cloud-init config to run on instance boot
    :return: The instance ID of the launched EC2 instance.
    """
    # If no SG was provided, create or get one that allows HTTP
    sg_id = ec2_config.security_group_id
    if not sg_id:
        sg_id = _create_or_fetch_security_group(ec2_client)
        ec2_config.security_group_id = sg_id
    print(f"[EC2EnvLauncher] Using security group: {sg_id}")

    # Prepare parameters
    params = {
        "ImageId": ec2_config.ami_id,
        "InstanceType": ec2_config.instance_type,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
    }
    if ec2_config.key_name:
        params["KeyName"] = ec2_config.key_name  # if you want SSH
    if ec2_config.subnet_id:
        params["SubnetId"] = ec2_config.subnet_id
    if user_data:
        params["UserData"] = user_data

    # Launch
    response = ec2_client.run_instances(**params)
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"[EC2EnvLauncher] Launched EC2 instance: {instance_id}")

    # Name the instance (optional)
    ec2_client.create_tags(
        Resources=[instance_id],
        Tags=[{"Key": "Name", "Value": ec2_config.instance_name}]
    )

    # Wait until it's running
    _wait_for_running(instance_id, ec2_config.region_name)
    return instance_id

def get_env_endpoint_impl(instance_id, region_name) -> Optional[str]:
    """
    Returns the public DNS of the launched instance as an HTTP endpoint,
    e.g., "http://ec2-xx-yy-zz.amazonaws.com".
    """
    if not instance_id:
        print("[EC2EnvLauncher] No instance launched yet.")
        return None

    ec2_resource = boto3.resource("ec2", region_name=region_name)
    instance = ec2_resource.Instance(instance_id)
    public_dns = instance.public_dns_name
    if public_dns:
        endpoint = f"http://{public_dns}"
        print(f"[EC2EnvLauncher] Environment endpoint: {endpoint}")
        return endpoint
    else:
        print("[EC2EnvLauncher] Instance does not have a public DNS name yet.")
        return None

def generate_user_data_script_impl(remote_image_uri: str) -> str:
    """
    Returns a shell script (User Data) that:
      1) Installs Docker (if not present) on the EC2 instance,
      2) Pulls the Docker image,
      3) Runs the container with a restart policy and ephemeral port mapping.

    This script is passed to `launch_ec2_instance(..., user_data=...)`.
    """
    user_data = f"""#!/bin/bash
set -e

echo "Installing Docker (if not present)..."
# Example for Amazon Linux / Red Hat-based AMIs:
yum update -y
yum install -y docker
service docker start

echo "Pulling Docker image: {remote_image_uri}"
docker pull {remote_image_uri}

echo "Running container from {remote_image_uri} with ephemeral ports"
# -d: Run container in background
# -P: Publish all exposed container ports to ephemeral host ports
# --restart unless-stopped: automatically restart container on reboot or crash
docker run -d -P --restart unless-stopped {remote_image_uri}
"""
    return user_data
