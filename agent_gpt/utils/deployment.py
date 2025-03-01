import os
import logging
from ..config.simulator import DockerfileConfig, K8SManifestConfig
from typing import List

logger = logging.getLogger(__name__)

def create_dockerfile(env_path: str, env_type, dockerfile: DockerfileConfig) -> str:
    """
    Generates a Dockerfile that packages the required application files and environment,
    based on the provided container configuration. The build context is set as the parent
    directory of the env_path, and the agent_gpt files are copied there so that the Dockerfile
    references them via relative paths.

    :param docker_config: DockerfileConfig object containing all deployment settings.
    :return: The path to the generated Dockerfile.
    """
    import shutil

    # Normalize env_path to use forward slashes.
    env_path = env_path.replace(os.sep, "/")
    additional_dependencies = dockerfile.additional_dependencies

    # Use the parent directory of env_path as the build context.
    # For example, if env_path is "C:/.../3DBallHard", then project_root becomes
    # "C:/.../unity_environments"
    project_root = os.path.dirname(os.path.abspath(env_path)).replace(os.sep, "/")
    # Compute the relative path from the project root to env_path.
    rel_env_path = os.path.relpath(env_path, project_root).replace(os.sep, "/")

    # Get the build files.
    # Expect build_files to now have relative paths with the prefix "agent_gpt/".
    build_files = get_build_files(env_type)    
    
    # Copy build files based on the paths returned by get_build_files.
    # Assume the source agent_gpt files are in the current working directory's "agent_gpt" folder.
    # They will be copied to the build context under "agent_gpt" (i.e. project_root/agent_gpt/).

    source_base = os.path.join(os.getcwd(), "agent_gpt")
    dest_base = os.path.join(project_root, "agent_gpt")

    for base_name, rel_path in build_files.items():
        src = os.path.join(source_base, rel_path)
        dest = os.path.join(dest_base, rel_path)
        # Ensure the destination directory exists.
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)
            logger.info(f"Copied {src} to {dest}")
        else:
            logger.warning(f"Source file {src} does not exist and cannot be copied.")

    # Place the Dockerfile in the build context (project_root).
    dockerfile_path = f"{project_root}/Dockerfile"
    logger.info(f"Creating Dockerfile at: {dockerfile_path}")
    logger.info(f" - Project root: {project_root}")
    logger.info(f" - Relative environment file path: {rel_env_path}")
    logger.info(f" - Env: {env_type}")

    # Internal container path where environment files are copied.
    cloud_import_path = "/app/env_files"

    with open(dockerfile_path, "w") as f:
        f.write("FROM python:3.9-slim\n\n")
        f.write("WORKDIR /app\n\n")

        # Copy agent_gpt project files.
        write_code_copy_instructions(f, build_files)

        if rel_env_path:
            f.write("# Copy environment files\n")
            f.write(f"RUN mkdir -p {cloud_import_path}\n")
            f.write(f"COPY {rel_env_path} {cloud_import_path}/\n\n")
        else:
            f.write("# No environment files to copy (env_path is None)\n")

        # Copy requirements and install dependencies.
        f.write("# Copy requirements.txt and install dependencies\n")
        # Assuming requirements.txt is inside the copied agent_gpt folder.
        f.write("COPY agent_gpt/requirements.txt /app/requirements.txt\n")
        f.write("RUN pip install --no-cache-dir --upgrade pip\n")
        f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")

        # Install any additional dependencies.
        for lib in additional_dependencies:
            f.write(f"RUN pip install --no-cache-dir {lib}\n")

        # Final command to run the environment server.
        f.write("# Final command to run the environment server\n")
        f.write(f'CMD ["python", "{build_files["entrypoint.py"]}", ')
        f.write(f'"{env_type}"]\n')

    logger.info(f"Done. Dockerfile written at: {dockerfile_path}")
    return dockerfile_path
        
def create_k8s_manifest(env_path: str, ports: List[int], 
                        k8s_config: K8SManifestConfig) -> str:
    """
    Generates a Kubernetes manifest YAML file for deploying the environment on EKS using PyYAML,
    based on the provided container configuration. The manifest is written into the build context,
    defined as the parent directory of the given env_path.
    
    :param env_path: The path to the environment files directory.
    :param k8s_config: K8SManifestConfig object containing all deployment settings.
    :return: The file path of the generated YAML manifest.
    """
    import os
    import yaml

    # Extract configuration values.
    image_name = k8s_config.image_name
    deployment_name = k8s_config.deployment_name
    container_ports = ports

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

    # Use the parent directory of env_path as the build context.
    project_root = os.path.dirname(os.path.abspath(env_path)).replace(os.sep, "/")
    file_path = f"{project_root}/{deployment_name}.yaml"
    
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
    entrypoint_file = "entrypoint.py"
    api_file = "env_host/api.py"
    gym_space_file = "utils/gym_space.py"
    data_converters_file = "utils/data_converters.py"

    if env == "gym":
        env_wrapper_file = "wrappers/gym_env.py"
    elif env == "unity":
        env_wrapper_file = "wrappers/unity_env.py"
    elif env == "custom":
        env_wrapper_file = "wrappers/custom_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env}'. Choose 'gym', 'unity', or 'custom'.")

    files = [entrypoint_file, api_file, gym_space_file, data_converters_file, env_wrapper_file]
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