
# env_hosting/cloud_host/cloud_env_launcher.py
from env_host.cloud.docker_builder import DockerEnvBuilder
from env_host.cloud.ec2_instance import EC2Instance
from config.aws_config import EC2Config

class CloudEnvLauncher(DockerEnvBuilder, EC2Instance):
    """
    A combined class that handles both Docker operations (build, push, user-data)
    and EC2 launching. We use multiple inheritance:
      - DockerEnv for Docker-related methods
      - EC2EnvLauncher for launching EC2 and retrieving endpoint
    """

    def __init__(self, ec2_config: EC2Config):
        # We must call the parent constructor of EC2EnvLauncher explicitly
        EC2Instance.__init__(self, ec2_config)
        # DockerEnv doesn't have a real constructor, so no need to call super(DockerEnv, self).__init__()
        
        
