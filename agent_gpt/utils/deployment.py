import os
import logging
import yaml
from ..config.container import DockerfileConfig, K8SManifestConfig

logger = logging.getLogger(__name__)

def create_dockerfile(docker_config: DockerfileConfig) -> str:
    """
    Generates a Dockerfile that packages the required application files and environment,
    based on the provided container configuration.

    :param container_config: DockerfileConfig object containing all deployment settings.
    :return: The path to the generated Dockerfile.
    """
    # Extract configuration values.
    env = docker_config.env
    env_path = docker_config.env_path
    env_id = docker_config.env_id
    entry_point = docker_config.entry_point
    additional_dependencies = docker_config.additional_dependencies
    container_ports = docker_config.container_ports

    # Define the Dockerfile location in the same directory as env_path.
    dockerfile_path = os.path.join(os.path.dirname(os.path.abspath(env_path)), "Dockerfile")
    logger.info(f"Creating Dockerfile at: {dockerfile_path}")
    logger.info(f" - Environment file path: {env_path}")
    logger.info(f" - Env: {env}")

    # Assume env_path is already in our build context.
    final_env_path = env_path

    # Internal container path where environment files are copied.
    cloud_import_path = "/app/env_files"

    build_files = get_build_files(env)

    with open(dockerfile_path, "w") as f:
        f.write("FROM python:3.9-slim\n\n")
        f.write("WORKDIR /app\n\n")

        # Expose container ports.
        for port in container_ports:
            f.write(f"EXPOSE {port}\n")
        f.write("\n")

        # Copy additional application files.
        write_code_copy_instructions(f, build_files)

        if final_env_path:
            f.write("# Copy environment files\n")
            f.write(f"RUN mkdir -p {cloud_import_path}\n")
            f.write(f"COPY {final_env_path} {cloud_import_path}/\n\n")
        else:
            f.write("# No environment files to copy (env_path is None)\n")

        # Copy requirements and install dependencies.
        f.write("# Copy requirements.txt and install dependencies\n")
        f.write("COPY requirements.txt /app/requirements.txt\n")
        f.write("RUN pip install --no-cache-dir --upgrade pip\n")
        f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")

        # Install any additional dependencies.
        for lib in additional_dependencies:
            f.write(f"RUN pip install --no-cache-dir {lib}\n")

        # Final command to run the environment server.
        f.write("# Final command to run the environment server\n")
        f.write(f'CMD ["python", "{build_files["entrypoint.py"]}", ')
        f.write(f'"{env}", "{env_id}", "{entry_point}"]\n')

    logger.info(f"Done. Dockerfile written at: {dockerfile_path}")
    return dockerfile_path


def create_k8s_manifest(k8s_config: K8SManifestConfig) -> str:
    """
    Generates a Kubernetes manifest YAML file for deploying the environment on EKS using PyYAML,
    based on the provided container configuration.

    :param container_config: K8SManifestConfig object containing all deployment settings.
    :return: The file path of the generated YAML manifest.
    """
    image_name = k8s_config.image_name
    deployment_name = k8s_config.deployment_name
    container_ports = k8s_config.container_ports

    # Define the Deployment spec.
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": deployment_name},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": deployment_name}},
            "template": {
                "metadata": {"labels": {"app": deployment_name}},
                "spec": {
                    "containers": [{
                        "name": deployment_name,
                        "image": image_name,
                        "ports": [{"containerPort": port} for port in container_ports]
                    }]
                }
            }
        }
    }

    # Define the Service spec.
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"{deployment_name}-svc"},
        "spec": {
            "type": "LoadBalancer",
            "selector": {"app": deployment_name},
            "ports": [
                {
                    "protocol": "TCP",
                    "port": port,
                    "targetPort": port
                } for port in container_ports
            ]
        }
    }

    manifest_yaml = f"{yaml.dump(deployment, sort_keys=False)}---\n{yaml.dump(service, sort_keys=False)}"
    file_path = f"{deployment_name}_k8s_manifest.yaml"
    with open(file_path, "w") as f:
        f.write(manifest_yaml)

    logger.info(f"Kubernetes manifest written to: {file_path}")
    return file_path


# ---------------- Helper Functions ----------------

def get_build_files(env: str) -> dict:
    """
    Returns a dictionary mapping file basenames to their paths required for the Docker build.

    :param env: The environment simulator ('gym', 'unity', or 'custom').
    :return: A dictionary of file paths needed for deployment.
    """
    entrypoint_file = "env_host/entrypoint.py"
    api_file = "env_host/api.py"
    utils_dir = "utils/"

    if env == "gym":
        env_wrapper_file = "wrappers/gym_env.py"
    elif env == "unity":
        env_wrapper_file = "wrappers/unity_env.py"
    elif env == "custom":
        env_wrapper_file = "wrappers/custom_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env}'. Choose 'gym', 'unity', or 'custom'.")

    files = [entrypoint_file, api_file, utils_dir, env_wrapper_file]
    return {os.path.basename(p.rstrip("/")): p for p in files}


def write_code_copy_instructions(f, build_files: dict):
    """
    Writes Docker COPY instructions for each file in build_files.

    :param f: The file handle for the Dockerfile.
    :param build_files: A dictionary mapping file basenames to file paths.
    """
    for base_name, rel_path in build_files.items():
        f.write(f"# Copy {base_name}\n")
        dir_part = os.path.dirname(rel_path.rstrip("/"))
        if dir_part:
            f.write(f"RUN mkdir -p /app/{dir_part}\n")
        f.write(f"COPY {rel_path} /app/{rel_path}\n\n")
