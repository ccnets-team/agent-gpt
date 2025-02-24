# src/env_host/cloud/cloud_launcher.py

import os
import shutil
import logging

class CloudEnvLauncher:
    """
    A class to generate a Dockerfile for hosting the environment on the cloud.
    
    This Dockerfile includes:
      * The necessary application files (e.g., serve.py, api.py)
      * The appropriate wrapper (gym_env.py, unity_env.py, or custom_env.py)
      * Additional files (e.g., utils/)
      * The specified environment file (which may include our extra dependency "env_api.py")
    
    Clients are expected to use the CLI to build, tag, push the Docker image, and deploy it (e.g., on EKS).
    """
    def __init__(self, env_simulator: str, env_id: str, env_file_path: str, global_image_name: str):
        """
        Initializes the CloudEnvLauncher.
        
        :param env_simulator: The RL environment simulator ('gym', 'unity', or 'custom').
        :param env_id: A unique ID or name for the environment.
        :param env_file_path: Path to the local environment file.
        :param global_image_name: Global name for the Docker image.
        """
        self.env_simulator = env_simulator
        self.env_id = env_id
        self.env_file_path = env_file_path
        self.global_image_name = global_image_name.lower()
        self.logger = logging.getLogger(__name__)
    
    def generate_dockerfile(
        self, 
        entry_point: str = None, 
        host: str = "0.0.0.0", 
        port: int = 80, 
        copy_env_file_if_outside: bool = False
    ) -> str:
        """
        Generates a Dockerfile at './Dockerfile' that packages the required application files and environment.
        
        :param entry_point: Optional entry point for the environment.
        :param host: Host address inside the container.
        :param port: Port number inside the container.
        :param copy_env_file_if_outside: If True, copies the environment file to './env_files' if it is outside the current directory.
        :return: The path to the generated Dockerfile.
        """
        dockerfile_path = "./Dockerfile"
        env_file_path = self.env_file_path
        
        self.logger.info(f"Creating Dockerfile at: {dockerfile_path}")
        self.logger.info(f" - Environment file path: {env_file_path}")
        self.logger.info(f" - Simulator: {self.env_simulator}")
        
        final_env_path = env_file_path
        if env_file_path and not is_in_current_directory(env_file_path) and copy_env_file_if_outside:
            self.logger.info(f"'{env_file_path}' is outside the current directory. Copying to './env_files/'.")
            safe_mkdir("env_files")
            env_basename = os.path.basename(env_file_path.rstrip("/"))
            final_env_path = os.path.join("env_files", env_basename)
            if os.path.isdir(env_file_path):
                shutil.copytree(env_file_path, final_env_path, dirs_exist_ok=True)
            else:
                shutil.copy2(env_file_path, final_env_path)
        else:
            self.logger.info("Environment file is in the current directory or copying is not required.")
        
        cloud_import_path = "/app/env_files"
        if entry_point:
            class_name = entry_point.split(":")[-1]
            cloud_entry_point = f"{cloud_import_path}:{class_name}"
        else:
            cloud_entry_point = cloud_import_path
        
        additional_files = get_additional_files(self.env_simulator)
        additional_libs = get_additional_libs(self.env_simulator, self.env_id)
        
        with open(dockerfile_path, "w") as f:
            f.write("FROM python:3.9-slim\n\n")
            f.write("WORKDIR /app\n\n")
            
            # Copy additional application files (e.g., serve.py, api.py, wrappers, utils/)
            write_code_copy_instructions(f, additional_files)
            
            if final_env_path:
                f.write("# Copy environment files\n")
                f.write(f"RUN mkdir -p {cloud_import_path}\n")
                f.write(f"COPY {final_env_path} {cloud_import_path}/\n\n")
            else:
                f.write("# No environment files to copy (env_file_path is None)\n")
            
            # Install dependencies from requirements.txt
            f.write("# Copy requirements.txt and install dependencies\n")
            f.write("COPY requirements.txt /app/requirements.txt\n")
            f.write("RUN pip install --no-cache-dir --upgrade pip\n")
            f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")
            
            # Install any additional libraries
            for lib in additional_libs:
                f.write(f"RUN pip install --no-cache-dir {lib}\n")
            
            # Final command to run the environment server
            f.write("# Final command to run the environment server\n")
            f.write(f'CMD ["python", "{additional_files["serve.py"]}", ')
            f.write(f'"{self.env_simulator}", "{self.env_id}", "{cloud_entry_point}", "{host}", "{port}"]\n')
        
        self.logger.info(f"Done. Dockerfile written at: {dockerfile_path}")
        return dockerfile_path

    def generate_k8s_manifest(
        self,
        deployment_name: str,
        container_port: int = 80,
        replicas: int = 1,
        service_type: str = "LoadBalancer"
    ) -> str:
        """
        Generates a Kubernetes manifest YAML file for deploying the environment on EKS.
        
        This manifest includes both a Deployment and a Service.
        
        :param deployment_name: Name to use for the Kubernetes Deployment.
        :param container_port: The port on which the container listens.
        :param replicas: Number of pod replicas.
        :param service_type: Kubernetes Service type (e.g., LoadBalancer, ClusterIP).
        :return: The file path of the generated YAML manifest.
        """
        # Use the global image name directly since tagging & pushing is handled via CLI.
        image = self.global_image_name
        
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
            image: {image}
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
        self.logger.info(f"Kubernetes manifest written to: {file_path}")
        return file_path

# Helper functions
def get_gymnasium_envs(categories=None):
    """
    Retrieves environment IDs from the Gymnasium registry filtered by categories.
    
    :param categories: List of categories (defaults to a set of common categories).
    :return: List of environment IDs.
    """
    from gymnasium import envs
    categories = categories or ["classic_control", "box2d", "toy_text", "mujoco", "phys2d", "tabular"]
    envs_by_category = {category: [] for category in categories}
    
    for env_spec in envs.registry.values():
        if isinstance(env_spec.entry_point, str):
            for category in categories:
                if category in env_spec.entry_point:
                    envs_by_category[category].append(env_spec.id)
                    break

    return [env_id for env_list in envs_by_category.values() for env_id in env_list]

def get_additional_files(env_simulator: str) -> dict:
    """
    Returns a dictionary mapping file basenames to their paths required for the Docker build.
    
    :param env_simulator: The environment simulator ('gym', 'unity', or 'custom').
    :return: Dictionary of file paths.
    """
    serve_file = "src/env_host/serve.py"
    api_file = "src/env_host/api.py"
    utils_file = "src/utils/"
    
    if env_simulator == "gym":
        env_wrapper_file = "src/wrappers/gym_env.py"
    elif env_simulator == "unity":
        env_wrapper_file = "src/wrappers/unity_env.py"
    elif env_simulator == "custom":
        env_wrapper_file = "src/wrappers/custom_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env_simulator}'. Choose 'gym', 'unity', or 'custom'.")
    
    files = [serve_file, api_file, utils_file, env_wrapper_file]
    return {os.path.basename(p.rstrip("/")): p for p in files}

def get_additional_libs(env_simulator: str, env_id: str):
    """
    Returns a list of additional libraries to install based on the simulator type and environment ID.
    
    :param env_simulator: The environment simulator ('gym', 'unity', or 'custom').
    :param env_id: The environment ID.
    :return: List of libraries for pip installation.
    """
    if env_simulator == "unity":
        return ["mlagents==0.30", "protobuf==3.20.0"]
    elif env_simulator == "gym":
        standard_env_ids = get_gymnasium_envs(["classic_control", "mujoco", "phys2d"])
        if env_id in standard_env_ids:
            return ["gymnasium[mujoco]"]
        return []
    elif env_simulator == "custom":
        return []
    else:
        raise ValueError(f"Unknown simulator '{env_simulator}'")

def write_code_copy_instructions(f, additional_files: dict):
    """
    Writes Docker COPY instructions for each provided file.
    
    :param f: File handle for the Dockerfile.
    :param additional_files: Dictionary mapping basenames to file paths.
    """
    for base_name, rel_path in additional_files.items():
        f.write(f"# Copy {base_name}\n")
        dir_part = os.path.dirname(rel_path.rstrip("/"))
        if dir_part:
            f.write(f"RUN mkdir -p /app/{dir_part}\n")
        f.write(f"COPY {rel_path} /app/{rel_path}\n\n")

def is_in_current_directory(path: str) -> bool:
    """
    Determines if a given path is within the current working directory.
    
    :param path: The path to check.
    :return: True if the path is within the current directory; otherwise, False.
    """
    current_dir = os.path.abspath(os.getcwd())
    target_abs = os.path.abspath(path)
    return os.path.commonprefix([current_dir, target_abs]) == current_dir

def safe_mkdir(path: str):
    """
    Creates the specified directory if it does not already exist.
    
    :param path: Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
