import boto3
from config.aws_config import EC2Config
from botocore.exceptions import ClientError
 
class LaunchEnvironment:
    def __init__(self, ec2_config: EC2Config):
        # Use 'ec2_config' consistently
        self.region_name = ec2_config.region_name
        self.security_group_name = ec2_config.security_group_name
        self.security_group_description = ec2_config.security_group_description
        self.vpc_id = ec2_config.vpc_id
        self.ami_id = ec2_config.ami_id
        self.instance_type = ec2_config.instance_type
        self.key_name = ec2_config.key_name
        self.subnet_id = ec2_config.subnet_id

    def create_security_group(self, my_ip: str) -> str:
        """
        Create or retrieve an existing security group that allows SSH
        from 'my_ip' (e.g., "1.2.3.4/32").
        """
        ec2 = boto3.client("ec2", region_name=self.region_name)
        sg_id = None

        # 1) Attempt to create the security group
        try:
            resp = ec2.create_security_group(
                GroupName=self.security_group_name,
                Description=self.security_group_description,
                VpcId=self.vpc_id
            )
            sg_id = resp["GroupId"]
            print(
                f"Created Security Group '{self.security_group_name}' with ID={sg_id}"
                f" in VPC={self.vpc_id}"
            )
        except ClientError as e:
            if "InvalidGroup.Duplicate" in str(e):
                # Already exists; retrieve its ID
                sg_info = ec2.describe_security_groups(
                    Filters=[{"Name": "group-name", "Values": [self.security_group_name]}]
                )
                sg_id = sg_info["SecurityGroups"][0]["GroupId"]
                print(
                    f"Security Group '{self.security_group_name}' already exists. "
                    f"ID={sg_id}"
                )
            else:
                raise e

        # 2) Authorize inbound SSH from my_ip on port 22
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": my_ip}]
                    }
                ]
            )
            print(f"Ingress SSH rule set for {my_ip}.")
        except ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                print("SSH rule already authorized.")
            else:
                raise e

        return sg_id

    def launch_ec2_instance(self, sg_id: str) -> str:
        """
        Launch an EC2 instance using the config’s AMI, instance type, and key pair.
        Uses the provided security group ID and optional subnet (if specified).
        Returns the instance ID.
        """
        ec2_client = boto3.client("ec2", region_name=self.region_name)
        params = {
            "ImageId": self.ami_id,
            "InstanceType": self.instance_type,
            "KeyName": self.key_name,
            "SecurityGroupIds": [sg_id],
            "MinCount": 1,
            "MaxCount": 1
        }
        if self.subnet_id:
            params["SubnetId"] = self.subnet_id

        # Run the instance
        response = ec2_client.run_instances(**params)
        instance_id = response["Instances"][0]["InstanceId"]
        print(f"Launched EC2 instance: {instance_id}")

        # Wait until instance is running
        ec2_resource = boto3.resource("ec2", region_name=self.region_name)
        instance = ec2_resource.Instance(instance_id)
        print(f"Waiting for instance {instance_id} to be in 'running' state...")
        instance.wait_until_running()
        instance.reload()

        print(f"Instance {instance_id} is now in state: {instance.state['Name']}")

        # Print the public IP or DNS
        if instance.public_ip_address:
            print(f"Public IP: {instance.public_ip_address}")
        if instance.public_dns_name:
            print(f"Public DNS: {instance.public_dns_name}")

        return instance_id

    def spin_up_ec2(self, my_ip: str) -> str:
        """
        High-level method: 
        1) Create or fetch the security group and open SSH.
        2) Launch the EC2 instance.
        3) Return the instance ID.
        """
        sg_id = self.create_security_group(my_ip=my_ip)
        instance_id = self.launch_ec2_instance(sg_id)
        return instance_id