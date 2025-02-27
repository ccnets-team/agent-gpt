import os
import logging

logger = logging.getLogger(__name__)

def generate_dockerfile(
    env: str,
    env_path: str,
    env_id: str = None,
    entry_point: str = None,
    additional_libs: list = []
) -> str:
    """
    Generates a Dockerfile that packages the required application files and environment.

    :param env: The RL environment simulator ('gym', 'unity', or 'custom').
    :param env_path: Path to the local environment file or directory.
    :param env_id: Optional environment identifier.
    :param entry_point: Optional entry point for the environment.
    :param additional_libs: A list of extra libraries to install.
    :return: The path to the generated Dockerfile.
    """
    # Define the Dockerfile location in the same directory as env_path:
    dockerfile_path = os.path.join(os.path.dirname(os.path.abspath(env_path)), "Dockerfile")
    logger.info(f"Creating Dockerfile at: {dockerfile_path}")
    logger.info(f" - Environment file path: {env_path}")
    logger.info(f" - Env: {env}")

    # In this simple case we assume env_path is already in our build context.
    final_env_path = env_path

    # This is the internal location inside the container where environment files are copied.
    cloud_import_path = "/app/env_files"

    additional_files = get_additional_files(env)

    with open(dockerfile_path, "w") as f:
        f.write("FROM python:3.9-slim\n\n")
        f.write("WORKDIR /app\n\n")

        # Copy additional application files (e.g., entrypoint.py, api.py, wrappers, utils/)
        write_code_copy_instructions(f, additional_files)

        if final_env_path:
            f.write("# Copy environment files\n")
            f.write(f"RUN mkdir -p {cloud_import_path}\n")
            f.write(f"COPY {final_env_path} {cloud_import_path}/\n\n")
        else:
            f.write("# No environment files to copy (env_path is None)\n")

        # Copy requirements and install dependencies
        f.write("# Copy requirements.txt and install dependencies\n")
        f.write("COPY requirements.txt /app/requirements.txt\n")
        f.write("RUN pip install --no-cache-dir --upgrade pip\n")
        f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")

        # Install any additional libraries
        for lib in additional_libs:
            f.write(f"RUN pip install --no-cache-dir {lib}\n")

        # Final command to run the environment server.
        # Note: In Kubernetes, your container must listen on the same port as defined in your manifest.
        f.write("# Final command to run the environment server\n")
        f.write(f'CMD ["python", "{additional_files["entrypoint.py"]}", ')
        f.write(f'"{env}", "{env_id}", "{entry_point}"]\n')

    logger.info(f"Done. Dockerfile written at: {dockerfile_path}")
    return dockerfile_path

def generate_kubernetes_manifest(
    image_name: str,
    deployment_name: str,
    container_port: int = 80,
    replicas: int = 1,
    service_type: str = "LoadBalancer"
) -> str:
    """
    Generates a Kubernetes manifest YAML file for deploying the environment on EKS.

    :param image_name: The Docker image name (with tag) to deploy.
    :param deployment_name: The name to use for the Kubernetes Deployment.
    :param container_port: The port on which the container listens (should match the Dockerfile CMD).
    :param replicas: The number of pod replicas.
    :param service_type: The Kubernetes Service type (e.g., LoadBalancer, ClusterIP).
    :return: The file path of the generated YAML manifest.
    """
    k8s_manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
      labels:
        app: {deployment_name}
    spec:
      containers:
      - name: {deployment_name}
        image: {image_name}
        ports:
        - containerPort: {container_port}
---
apiVersion: v1
kind: Service
metadata:
  name: {deployment_name}-svc
spec:
  type: {service_type}
  selector:
    app: {deployment_name}
  ports:
  - protocol: TCP
    port: {container_port}
    targetPort: {container_port}
"""
    file_path = f"{deployment_name}_k8s_manifest.yaml"
    with open(file_path, "w") as f:
        f.write(k8s_manifest)
    logger.info(f"Kubernetes manifest written to: {file_path}")
    return file_path

# ---------------- Helper Functions ----------------

def get_additional_files(env: str) -> dict:
    """
    Returns a dictionary mapping file basenames to their paths required for the Docker build.

    :param env: The environment simulator ('gym', 'unity', or 'custom').
    :return: A dictionary of file paths.
    """
    serve_file = "env_host/entrypoint.py"
    api_file = "env_host/api.py"
    utils_file = "utils/"

    if env == "gym":
        env_wrapper_file = "wrappers/gym_env.py"
    elif env == "unity":
        env_wrapper_file = "wrappers/unity_env.py"
    elif env == "custom":
        env_wrapper_file = "wrappers/custom_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env}'. Choose 'gym', 'unity', or 'custom'.")

    files = [serve_file, api_file, utils_file, env_wrapper_file]
    return {os.path.basename(p.rstrip("/")): p for p in files}

def write_code_copy_instructions(f, additional_files: dict):
    """
    Writes Docker COPY instructions for each file in additional_files.

    :param f: The file handle for the Dockerfile.
    :param additional_files: A dictionary mapping file basenames to file paths.
    """
    for base_name, rel_path in additional_files.items():
        f.write(f"# Copy {base_name}\n")
        dir_part = os.path.dirname(rel_path.rstrip("/"))
        if dir_part:
            f.write(f"RUN mkdir -p /app/{dir_part}\n")
        f.write(f"COPY {rel_path} /app/{rel_path}\n\n")
