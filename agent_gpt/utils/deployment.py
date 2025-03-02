import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def create_dockerfile(env_type: str, env_path: str, additional_dependencies: List) -> str:
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

    # Use the parent directory of env_path as the build context.
    # For example, if env_path is "C:/.../3DBallHard", then project_root becomes
    project_root = os.path.dirname(os.path.abspath(env_path)).replace(os.sep, "/")
    # Compute the relative path from the project root to env_path.
    rel_env_path = os.path.relpath(env_path, project_root).replace(os.sep, "/")

    # Get the build files.
    # Expect build_files to now have relative paths with the prefix "agent_gpt/".
    build_files = get_build_files(env_type)    
    
    # Copy build files based on the paths returned by get_build_files.
    # Assume the source agent_gpt files are in the current working directory's "agent_gpt" folder.
    # They will be copied to the build context under "agent_gpt" (i.e. project_root/agent_gpt/).

    # If using Unity, ensure the required Unity-specific dependencies are included.
    if env_type.lower() == "unity":
        required_unity_deps = ["mlagents_envs==0.30.0", "protobuf==3.20.0"]
        for dep in required_unity_deps:
            if dep not in additional_dependencies:
                logger.warning(f"Unity required dependency '{dep}' is missing. Adding it automatically.")
                additional_dependencies.append(dep)

    source_base = os.path.join(os.getcwd(), "agent_gpt")
    dest_base = os.path.join(project_root, "agent_gpt")
                
    for base_name, rel_path in build_files.items():
        src = os.path.join(source_base, rel_path).replace(os.sep, "/")
        dest = os.path.join(dest_base, rel_path).replace(os.sep, "/")
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

    requirements_src = os.path.join(os.getcwd(), "requirements.txt").replace(os.sep, "/")
    requirements_dest = os.path.join(project_root, "requirements.txt").replace(os.sep, "/")
    if os.path.exists(requirements_src):
        shutil.copy2(requirements_src, requirements_dest)
        logger.info(f"Copied {requirements_src} to {requirements_dest}")
    else:
        logger.warning(f"requirements.txt not found at {requirements_src}")
        
    # Place the Dockerfile in the build context (project_root).
    dockerfile_path = f"{project_root}/Dockerfile"
    logger.info(f"Creating Dockerfile at: {dockerfile_path}")
    logger.info(f" - Project root: {project_root}")
    logger.info(f" - Relative environment file path: {rel_env_path}")
    logger.info(f" - Env Type: {env_type}")

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
        f.write("COPY requirements.txt /app/requirements.txt\n")
        f.write("RUN pip install --no-cache-dir --upgrade pip\n")
        f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")

        # Install any additional dependencies.
        for lib in additional_dependencies:
            f.write(f"RUN pip install --no-cache-dir {lib}\n")

        # Final command to run the environment server.
        f.write("# Final command to run the environment server\n")
        f.write(f'CMD ["python", "agent_gpt/{build_files["entrypoint.py"]}", ')
        f.write(f'"{env_type}"]\n')

    logger.info(f"Done. Dockerfile written at: {dockerfile_path}")
    return dockerfile_path

def get_build_files(env: str) -> dict:
    """
    Returns a dictionary mapping file basenames to their paths required for the Docker build.

    :param env: The environment simulator ('gym', 'unity', or 'custom').
    :return: A dictionary of file paths needed for deployment.
    """
    entrypoint_file = "entrypoint.py"
    api_file = "env_host/env_api.py"
    data_converters_file = "utils/conversion_utils.py"

    if env == "gym":
        env_wrapper_file = "wrappers/gym_env.py"
    elif env == "unity":
        env_wrapper_file = "wrappers/unity_env.py"
    elif env == "custom":
        env_wrapper_file = "wrappers/custom_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env}'. Choose 'gym', 'unity', or 'custom'.")

    files = [entrypoint_file, api_file, data_converters_file, env_wrapper_file]
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
        f.write(f"COPY agent_gpt/{rel_path} /app/agent_gpt/{rel_path}\n\n")            
        
# ---------------- Kerbernetes Deployment & Service ----------------
from kubernetes import client

def deploy_eks_simulator(deployment_name, image_uri, ports, namespace="default"):
    """
    Create a Kubernetes Deployment with the given parameters.
    
    :param deployment_name: Name of the Deployment.
    :param image_uri: Fully qualified image URI (e.g., from ECR).
    :param ports: List of container ports to expose.
    :param namespace: Kubernetes namespace (default is "default").
    """
    # Define container ports based on the provided list
    container_ports = [client.V1ContainerPort(container_port=p) for p in ports]

    # Define the container with the image and ports
    container = client.V1Container(
        name=deployment_name,
        image=image_uri,
        ports=container_ports
    )

    # Create the pod template with a label matching the deployment name
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
        spec=client.V1PodSpec(containers=[container])
    )

    # Define the deployment spec with one replica
    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels={"app": deployment_name}),
        template=template
    )

    # Create the Deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=spec
    )

    # Use the AppsV1Api to create the deployment in the specified namespace
    apps_v1 = client.AppsV1Api()
    resp = apps_v1.create_namespaced_deployment(
        body=deployment,
        namespace=namespace
    )
    print(f"Deployment '{resp.metadata.name}' created.")

def service_eks_simulator(deployment_name, ports, namespace="default"):
    """
    Create a Kubernetes Service that maps external ports to container ports.
    
    :param deployment_name: Name of the associated deployment.
    :param ports: List of ports to expose.
    :param namespace: Kubernetes namespace (default is "default").
    """
    # Define a list of ServicePort objects mapping each port directly
    service_ports = [client.V1ServicePort(protocol="TCP", port=p, target_port=p) for p in ports]

    # Create a Service spec with type LoadBalancer
    service_spec = client.V1ServiceSpec(
        selector={"app": deployment_name},
        ports=service_ports,
        type="LoadBalancer"
    )

    # Create the Service object with a name derived from the deployment
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=f"{deployment_name}-service"),
        spec=service_spec
    )

    # Use the CoreV1Api to create the service in the specified namespace
    core_v1 = client.CoreV1Api()
    resp = core_v1.create_namespaced_service(
        body=service,
        namespace=namespace
    )
    print(f"Service '{resp.metadata.name}' created.")
    
# ---------------- App Runner Deployment ----------------    
    
import boto3
from ..config.simulator import SimulatorConfig


def simulate_app_runner(simulator_config: SimulatorConfig, region, cpu='1024', memory='2048'):
    service_name = simulator_config.container.deployment_name
    image_uri = simulator_config.container.image_uri
    # Use the first port since App Runner supports a single port.
    port = simulator_config.ports[0] if simulator_config.ports else 80

    # Ensure you are using the correct region where your ECR image resides.
    apprunner = boto3.client('apprunner', region_name=region)
    
    try:
        response = apprunner.create_service(
            ServiceName=service_name,
            SourceConfiguration={
                'ImageRepository': {
                    'ImageIdentifier': image_uri,
                    'ImageRepositoryType': 'ECR',
                    'ImageConfiguration': {
                        'Port': str(port)
                    }
                },
                'AutoDeploymentsEnabled': True
            },
            InstanceConfiguration={
                'Cpu': cpu,
                'Memory': memory
            },
            HealthCheckConfiguration={
                'Protocol': 'HTTP',
                'Path': '/',
                'Interval': 10,
                'Timeout': 5,
                'HealthyThreshold': 1,
                'UnhealthyThreshold': 5,
            }
        )
        logger.info(f"App Runner service '{service_name}' created successfully.")
    except Exception as e:
        logger.error(f"Error creating App Runner service '{service_name}': {e}")
        raise
    return response