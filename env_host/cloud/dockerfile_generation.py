# env_host/cloud/docker_builder.py
import os

def get_gymnasium_envs(categories=None):
    from gymnasium import envs
    """
    Retrieves environment IDs grouped by specified categories based on entry points in the Gymnasium registry.
    
    Returns:
        list: A list of all environment IDs in the specified categories.
    """
    categories = categories or ["classic_control", "box2d", "toy_text", "mujoco", "phys2d", "tabular"]
    envs_by_category = {category: [] for category in categories}
    
    for env_spec in envs.registry.values():
        if isinstance(env_spec.entry_point, str):
            for category in categories:
                if category in env_spec.entry_point:
                    envs_by_category[category].append(env_spec.id)
                    break  # Stop after matching this category

    # Flatten all categories into a single list of env IDs
    all_registered_envs = [env_id for env_list in envs_by_category.values() for env_id in env_list]
    return all_registered_envs

def get_additional_files(env_simulator: str) -> dict:
    """
    Returns a dict where:
        key = the full path relative to the build context
        value = the filename (basename only)
    """
    serve_file = "env_host/serve.py"
    api_file   = "env_host/api.py"
    utils_file = "utils/"

    if env_simulator == "gym":
        env_wrapper_file = "env_host/wrappers/gym_env.py"
    elif env_simulator == "unity":
        env_wrapper_file = "env_host/wrappers/unity_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env_simulator}'. Choose 'gym' or 'unity'.")

    # E.g. {
    #    "env_host/serve.py": "serve.py",
    #    "env_host/api.py": "api.py",
    #    "env_host/wrappers/gym_env.py": "gym_env.py"
    # }
    files = [serve_file, api_file, utils_file, env_wrapper_file]
    return {os.path.basename(p): p for p in files}

def get_additional_libs(env_simulator: str, env_id: str):
    """
    Returns a list of extra pip install commands, if needed, based on env_simulator and env_id.
    """
    if env_simulator == "unity":
        # Example: ML-Agents + older protobuf requirement
        return ["mlagents==0.30", "protobuf==3.20.0"]
    elif env_simulator == "gym":
        # If env_id is a standard Gymnasium environment, we add gymnasium[mujoco]
        standard_env_ids = get_gymnasium_envs(["classic_control", "mujoco", "phys2d"])
        if env_id in standard_env_ids:
            return ["gymnasium[mujoco]"]
        return []
    else:
        raise ValueError(f"Unknown simulator '{env_simulator}'")


def write_code_copy_instructions(f, additional_files: dict):
    """
    Writes Docker COPY instructions for each full_path -> basename in `include_dict`.
    Example:
        # Copy serve.py
        RUN mkdir -p /app/env_host
        COPY env_host/serve.py /app/env_host/serve.py
    """
    for base_name, full_path in additional_files.items():
        f.write(f"# Copy {base_name}\n")
        dir_part = os.path.dirname(full_path)
        # Make sure the directory structure exists inside /app
        if dir_part:
            f.write(f"RUN mkdir -p /app/{dir_part}\n")
        f.write(f"COPY {full_path} /app/{full_path}\n\n")
    

def generate_dockerfile_impl(env_simulator: str,
                        env_id: str,
                        env_file_path: str,
                        host: str = "0.0.0.0",
                        port: int = 80,
                        dockerfile_path: str = "./Dockerfile") -> str:
    """
    Generates a Dockerfile that only copies:
      1) serve.py
      2) api.py
      3) gym_env.py or unity_env.py (depending on env_simulator)
      4) env_file_path (e.g., '3DBallHard' or some Unity environment folder)
    ... into the container, ignoring other files.
    """
    print(f"[generate_dockerfile] Creating Dockerfile at: {dockerfile_path}")
    print(f" - Environment file path to copy: {env_file_path}")
    print(f" - Simulator: {env_simulator}")

    additional_files = get_additional_files(env_simulator)
    additional_libs = get_additional_libs(env_simulator, env_id)

    with open(dockerfile_path, "w") as f:
        # Base image
        f.write("FROM python:3.9-slim\n\n")
        f.write("WORKDIR /app\n\n")

        # Copy only specific files (serve.py, api.py, gym_env/unity_env)
        write_code_copy_instructions(f, additional_files)

        # Copy environment files/folder (Unity build or local env data)
        if env_file_path is not None:
            f.write("# Copy environment files\n")
            f.write("RUN mkdir -p /app/env_files\n")
            f.write(f"COPY {env_file_path} /app/env_files/\n\n")
        else:
            f.write("# No environment files to copy (env_file_path=None)\n")
            
        # Copy and install dependencies if present
        f.write("# Copy requirements.txt and install dependencies\n")
        f.write("COPY requirements.txt /app/requirements.txt\n")
        f.write("RUN pip install --no-cache-dir --upgrade pip\n")
        f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")

        # Additional libraries for certain simulators/environments
        for lib in additional_libs:
            f.write(f"RUN pip install --no-cache-dir {lib}\n")

        # Final CMD: run serve.py with the correct simulator argument
        # We know serve.py is at "env_host/serve.py"
        f.write(f'CMD ["python", "{additional_files["serve.py"]}", "{env_simulator}", "{host}", "{port}"]\n')

    print("[generate_dockerfile] Done.")
    return dockerfile_path