# env_hosting/cloud_host/cloud_launcher.py
from config.aws_config import EC2Config
from env_host.cloud.dockerfile_generator import generate_dockerfile_impl
from env_host.cloud.docker_env_builder import (
    detect_docker_cmd,
    build_docker_image_impl,
    tag_docker_image_impl,
    push_docker_image_impl
)
from env_host.cloud.ec2_env_launcher import (
    generate_user_data_script_impl,
    launch_ec2_instance_impl,
    get_env_endpoint_impl
)
import boto3
import logging
from typing import Optional

class CloudEnvLauncher:
    """
    A class that orchestrates the process of:
      * Generating a Dockerfile
      * Building and pushing a Docker image to ECR
      * Launching an EC2 instance to run the environment
      * Retrieving the environment endpoint
    """

    def __init__(
        self,
        env_simulator: str,
        env_id: str,
        env_file_path: str,
        global_image_name: str,
        ecr_registry: str,
        ec2_config: EC2Config
    ):
        """
        Initializes the CloudEnvLauncher.

        :param env_simulator: An object representing the RL environment simulator.
        :param env_id: A unique ID or name for the environment.
        :param env_file_path: The path to the local environment file.
        :param global_image_name: global name for the Docker/ECR image.
        :param ec2_config: An EC2Config object containing AWS EC2 configuration.
        :param ecr_registry: The ECR registry URI, e.g., "123456789012.dkr.ecr.us-east-1.amazonaws.com".
        """
        self.ec2_config: EC2Config = ec2_config
        self.region_name: str = ec2_config.region_name
        self.env_simulator = env_simulator
        self.env_id: str = env_id
        self.global_image_name: str = global_image_name.lower()
        self.ecr_registry: str = ecr_registry
        self.env_file_path: str = env_file_path

        self.os_name, self.docker_cmd = detect_docker_cmd()
        self.ec2_client = boto3.client("ec2", region_name=self.region_name)

        self.dockerfile_path = "./Dockerfile"
        
        self.instance_id: Optional[str] = None
        self.remote_image_uri: Optional[str] = None

        self.logger = logging.getLogger(__name__)

    def launch_remote_env(self, ensure_ecr_login: bool = True, ensure_ecr_repo: bool = True) -> str:
        """
        High-level method that orchestrates Docker/ECR/EC2 flow:

        1. Generates a Dockerfile
        2. Builds the Docker image locally
        3. Tags & pushes the image to ECR
        4. Launches an EC2 instance with the new image
        5. Retrieves and returns the environment endpoint

        :param ensure_ecr_repo: Whether to ensure the ECR repository is created if it doesn't exist.
        :return: The endpoint of the launched environment.
        """
        self.generate_dockerfile(env_file_path=self.env_file_path, dockerfile_path=self.dockerfile_path)
        self.build_docker_image(docker_image_name=self.global_image_name, dockerfile_path=self.dockerfile_path)
        self.tag_docker_image(ecr_registry=self.ecr_registry, local_image_name=self.global_image_name)
        self.push_docker_image(
            ecr_registry=self.ecr_registry,
            local_image_name=self.global_image_name,
            ensure_ecr_login=ensure_ecr_login,
            ensure_ecr_repo=ensure_ecr_repo
        )
        self.launch_ec2_instance()
        endpoint = self.get_env_endpoint()
        return endpoint
                
    def generate_dockerfile(
        self, 
        env_file_path: Optional[str] = None, 
        dockerfile_path: str = "./Dockerfile"
    ) -> str:
        """
        Generates a Dockerfile from the specified environment file.

        :param env_file_path: Path to the local environment file. If None, defaults to self.env_file_path.
        :param dockerfile_path: The output path for the Dockerfile. Defaults to "./Dockerfile".
        :return: The path to the generated Dockerfile.
        """
        env_file_path = env_file_path or self.env_file_path
        dockerfile_path = dockerfile_path or self.dockerfile_path
        return generate_dockerfile_impl(
            self.env_simulator,
            self.env_id,
            local_import_path = env_file_path,
            dockerfile_path=dockerfile_path
        )

    def build_docker_image(
        self,
        docker_image_name: Optional[str] = None,
        dockerfile_path: str = "./Dockerfile"
    ) -> None:
        """
        Builds a Docker image using a specified Dockerfile.

        :param docker_image_name: The local name for the Docker image. If None, defaults to self.docker_image_name.
        :param dockerfile_path: The path to the Dockerfile. Defaults to "./Dockerfile".
        :raises Exception: Raises an exception if the Docker build command fails.
        """
        dockerfile_path = dockerfile_path or self.dockerfile_path
        docker_image_name = docker_image_name or self.global_image_name
        docker_image_name = docker_image_name.lower()

        self.logger.info(f"Building Docker image '{docker_image_name}' from '{dockerfile_path}'...")
        
        try:
            build_docker_image_impl(self.docker_cmd, docker_image_name, dockerfile_path)
            self.logger.info(f"Successfully built image '{docker_image_name}'")
        except Exception as e:
            self.logger.error(f"Error building Docker image '{docker_image_name}': {str(e)}")
            raise

    def tag_docker_image(
        self,
        ecr_registry: Optional[str] = None,
        local_image_name: Optional[str] = None,
        remote_image_name: Optional[str] = None
    ) -> None:
        """
        Tags a locally built Docker image to prepare for pushing to ECR.

        :param ecr_registry: The ECR registry URI. If None, uses self.ecr_registry.
        :param local_image_name: The local Docker image name. If None, uses self.docker_image_name.
        :param remote_image_name: The remote Docker image name. If None, uses self.docker_image_name.
        """
        ecr_registry = ecr_registry or self.ecr_registry
        local_image_name = (local_image_name or self.global_image_name).lower()
        remote_image_name = (remote_image_name or self.global_image_name).lower()

        self.logger.info(f"Tagging local image '{local_image_name}' for registry '{ecr_registry}'...")
        tag_docker_image_impl(self.docker_cmd, local_image_name, remote_image_name)

    def push_docker_image(
        self,
        ecr_registry: Optional[str] = None,
        local_image_name: Optional[str] = None,
        remote_image_name: Optional[str] = None,
        ensure_ecr_login: bool = False,
        ensure_ecr_repo: bool = False
    ) -> str:
        """
        Pushes a Docker image to an ECR repository.

        :param ecr_registry: The ECR registry URI. If None, uses self.ecr_registry.
        :param local_image_name: The local Docker image name. If None, uses self.docker_image_name.
        :param remote_image_name: The remote Docker image name. If None, uses self.docker_image_name.
        :param ensure_ecr_login: Whether to perform 'docker login' to ECR before pushing.
        :param ensure_ecr_repo: Whether to create the ECR repo if it does not exist.
        :return: The URI of the pushed Docker image in ECR.
        """
        ecr_registry = ecr_registry or self.ecr_registry
        local_image_name = (local_image_name or self.global_image_name).lower()
        remote_image_name = (remote_image_name or self.global_image_name).lower()

        self.logger.info(f"Pushing image '{local_image_name}' to ECR registry '{ecr_registry}'...")
        self.remote_image_uri = push_docker_image_impl(
            self.docker_cmd,
            ecr_registry,
            local_image_name,
            remote_image_name,
            self.region_name,
            ensure_ecr_login,
            ensure_ecr_repo
        )
        self.logger.info(f"Successfully pushed image: {self.remote_image_uri}")
        return self.remote_image_uri

    def launch_ec2_instance(self, remote_image_uri: Optional[str] = None) -> str:
        """
        Launches an EC2 instance that runs the specified Docker image via user data.

        :param remote_image_uri: The URI of the Docker image in ECR. If None, uses self.remote_image_uri.
        :return: The AWS EC2 instance ID.
        """
        remote_image_uri = remote_image_uri or self.remote_image_uri
        user_data = generate_user_data_script_impl(remote_image_uri)

        self.logger.info(f"Launching EC2 instance in region '{self.region_name}' with image '{remote_image_uri}'...")
        self.instance_id = launch_ec2_instance_impl(self.ec2_config, self.ec2_client, user_data)
        self.logger.info(f"EC2 instance launched with ID: {self.instance_id}")
        return self.instance_id

    def get_env_endpoint(self, instance_id) -> str:
        """
        Retrieves the environment endpoint (public DNS or IP) from the launched EC2 instance.

        :return: The endpoint (public DNS or IP) of the environment.
        """
        if instance_id:
          self.instance_id = instance_id
        endpoint = get_env_endpoint_impl(self.instance_id, self.region_name)
        self.logger.info(f"Retrieved endpoint for instance '{self.instance_id}': {endpoint}")
        return endpoint
