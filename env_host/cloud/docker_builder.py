import platform
import subprocess
import os

class DockerEnvBuilder:
    """
    Manages Docker-related activities: generating Dockerfiles, building, pushing images, 
    and creating user-data scripts that pull/run the container on EC2.
    """

    def __init__(self):
        # Detect operating system
        self.os_name = platform.system().lower()
        # On Windows, you may need 'docker.exe' explicitly; on Linux/macOS, usually 'docker'
        self.docker_cmd = "docker.exe" if "windows" in self.os_name else "docker"

    def generate_docker_file(self, env_simulator: str, env_path: str) -> str:
        """
        Creates (or modifies) a Dockerfile in `env_path` based on the env_simulator (e.g., Unity, Gym).
        Returns the path to the Dockerfile.
        """
        dockerfile_path = os.path.join(env_path, "Dockerfile")
        print(f"[DockerEnvBuilder] Generating Dockerfile at: {dockerfile_path} for simulator: {env_simulator}")

        # A minimal example Dockerfile—customize as needed
        with open(dockerfile_path, "w") as f:
            f.write("FROM python:3.9-slim\n")
            f.write("WORKDIR /app\n")
            f.write("# Install Unity or Gym dependencies as needed...\n")
            if env_simulator == "unity":
                f.write("# e.g., apt-get update && apt-get install -y libs...\n")
            elif env_simulator == "gym":
                f.write("# e.g., pip install gymnasium\n")
            f.write('CMD ["python", "-m", "http.server", "80"]\n')  # Basic HTTP server for demonstration

        return dockerfile_path

    def build_docker_image(self, dockerfile_path: str, local_image_name: str):
        """
        Builds a Docker image using the specified Dockerfile and tags it with `local_image_name`.
        """
        # The directory containing the Dockerfile is typically the "context" for `docker build`
        context_dir = os.path.dirname(dockerfile_path)

        build_command = [
            self.docker_cmd,
            "build",
            "-t",
            local_image_name,
            "-f",
            dockerfile_path,
            context_dir
        ]

        print(f"[DockerEnvBuilder] Building Docker image '{local_image_name}' "
              f"from Dockerfile: {dockerfile_path} (context='{context_dir}')")

        try:
            subprocess.run(build_command, check=True)
            print(f"[DockerEnvBuilder] Successfully built image '{local_image_name}'")
        except subprocess.CalledProcessError as e:
            print(f"[DockerEnvBuilder] ERROR building image '{local_image_name}': {e}")
            raise

    def push_docker_image(self, local_image_name: str, remote_image_name: str, ecr_registry: str):
        """
        Pushes a local Docker image to a remote registry (like ECR).
        Steps typically include:
          1) docker tag {local_image_name} {ecr_registry}/{remote_image_name}
          2) docker push {ecr_registry}/{remote_image_name}
        """
        full_remote_name = f"{ecr_registry}/{remote_image_name}"

        # 1) Tag the local image
        tag_command = [
            self.docker_cmd,
            "tag",
            local_image_name,
            full_remote_name
        ]
        print(f"[DockerEnvBuilder] Tagging '{local_image_name}' as '{full_remote_name}'")

        try:
            subprocess.run(tag_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[DockerEnvBuilder] ERROR tagging image '{local_image_name}': {e}")
            raise

        # 2) Push the tagged image
        push_command = [
            self.docker_cmd,
            "push",
            full_remote_name
        ]
        print(f"[DockerEnvBuilder] Pushing image to '{full_remote_name}'")

        try:
            subprocess.run(push_command, check=True)
            print(f"[DockerEnvBuilder] Successfully pushed image to '{full_remote_name}'")
        except subprocess.CalledProcessError as e:
            print(f"[DockerEnvBuilder] ERROR pushing image '{full_remote_name}': {e}")
            raise

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
