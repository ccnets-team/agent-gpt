import os
import shutil
        
def get_gymnasium_envs(categories=None):
    """
    Retrieves environment IDs grouped by specified categories 
    based on entry points in the Gymnasium registry.
    
    Returns:
        list of str: All environment IDs in the specified categories.
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

    # Flatten all categories into a single list
    all_registered_envs = [env_id for env_list in envs_by_category.values() for env_id in env_list]
    return all_registered_envs

def get_additional_files(env_simulator: str) -> dict:
    """
    Returns a dict where the key is the basename and the value is the full 
    path relative to the build context. For example:
      {
        "serve.py": "src/env_host/serve.py",
        "api.py": "src/env_host/api.py",
        "utils": "utils/",
        "gym_env.py": "src/env_host/wrappers/gym_env.py"
      }
    """
    serve_file = "src/env_host/serve.py"
    api_file   = "src/env_host/api.py"
    utils_file = "utils/"

    if env_simulator == "gym":
        env_wrapper_file = "wrappers/gym_env.py"
    elif env_simulator == "unity":
        env_wrapper_file = "wrappers/unity_env.py"
    elif env_simulator == "custom":
        env_wrapper_file = "wrappers/custom_env.py"
    else:
        raise ValueError(f"Unknown simulator '{env_simulator}'. Choose 'gym' or 'unity' or 'custom'.")

    files = [serve_file, api_file, utils_file, env_wrapper_file]
    return {os.path.basename(p.rstrip("/")): p for p in files}

def get_additional_libs(env_simulator: str, env_id: str):
    """
    Returns a list of extra pip install commands, if needed, 
    based on env_simulator and env_id.
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
    Writes Docker COPY instructions for each {basename -> path}.
    Example:
        # Copy serve.py
        RUN mkdir -p /app/env_host
        COPY src/env_host/serve.py /app/env_host/serve.py
    """
    for base_name, rel_path in additional_files.items():
        f.write(f"# Copy {base_name}\n")
        dir_part = os.path.dirname(rel_path.rstrip("/"))
        if dir_part:
            f.write(f"RUN mkdir -p /app/{dir_part}\n")
        f.write(f"COPY {rel_path} /app/{rel_path}\n\n")

def is_in_current_directory(path: str) -> bool:
    """
    Returns True if 'path' (made absolute) is a subpath of the current working directory.
    """
    current_dir = os.path.abspath(os.getcwd())
    target_abs  = os.path.abspath(path)
    return os.path.commonprefix([current_dir, target_abs]) == current_dir

def safe_mkdir(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def generate_dockerfile_impl(env_simulator: str,
                             env_id: str,
                             env_file_path: str,
                             entry_point: str = None,
                             host: str = "0.0.0.0",
                             port: int = 80,
                             copy_env_file_if_outside: bool = False) -> str:
    """
    Generates a Dockerfile (always at ./Dockerfile) that copies:
      1) serve.py, api.py
      2) wrappers (gym_env.py, unity_env.py, or custom_env.py)
      3) utils/
      4) env_file_path (if provided)
      
    If `copy_env_file_if_outside_cwd` is True and env_file_path is 
    outside the current directory, it is copied into ./env_files/<basename>.
    """
    dockerfile_path = "./Dockerfile"
    print(f"[generate_dockerfile] Creating Dockerfile at: {dockerfile_path}")
    print(f" - Environment file path to copy: {env_file_path}")
    print(f" - Simulator: {env_simulator}")

    final_env_path = env_file_path  # the path we'll reference in the Dockerfile

    # If env_file_path is provided, outside the current directory, and user wants to copy it:
    if env_file_path and not is_in_current_directory(env_file_path) and copy_env_file_if_outside:
        print(f"[*] '{env_file_path}' is outside the current directory. Copying to './env_files/'.")
        safe_mkdir("env_files")
        env_basename = os.path.basename(env_file_path.rstrip("/"))
        final_env_path = os.path.join("env_files", env_basename)

        if os.path.isdir(env_file_path):
            shutil.copytree(env_file_path, final_env_path, dirs_exist_ok=True)
        else:
            shutil.copy2(env_file_path, final_env_path)
    else:
        print("[*] Environment file is already in the current directory or copying not required.")

    # Figure out how we reference it inside the container
    cloud_import_path = "/app/env_files"
    if entry_point:
        class_name = entry_point.split(":")[-1]
        cloud_entry_point = f"{cloud_import_path}:{class_name}"
    else:
        cloud_entry_point = cloud_import_path

    # Gather files and libs
    additional_files = get_additional_files(env_simulator)
    additional_libs = get_additional_libs(env_simulator, env_id)

    # Write Dockerfile
    with open(dockerfile_path, "w") as f:
        f.write("FROM python:3.9-slim\n\n")
        f.write("WORKDIR /app\n\n")

        # Copy serve.py, api.py, wrappers, utils/
        write_code_copy_instructions(f, additional_files)

        # Copy env file/folder if final_env_path is set
        if final_env_path:
            f.write("# Copy environment files\n")
            f.write(f"RUN mkdir -p {cloud_import_path}\n")
            f.write(f"COPY {final_env_path} {cloud_import_path}/\n\n")
        else:
            f.write("# No environment files to copy (env_file_path=None)\n")

        # Requirements + pip install
        f.write("# Copy requirements.txt and install dependencies\n")
        f.write("COPY requirements.txt /app/requirements.txt\n")
        f.write("RUN pip install --no-cache-dir --upgrade pip\n")
        f.write("RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi\n\n")

        # Additional libs
        for lib in additional_libs:
            f.write(f"RUN pip install --no-cache-dir {lib}\n")

        # Final CMD
        f.write("# Final command to run the environment server\n")
        f.write(f'CMD ["python", "{additional_files["serve.py"]}", ')
        f.write(f'"{env_simulator}", "{env_id}", "{cloud_entry_point}", "{host}", "{port}"]\n')

    print(f"[generate_dockerfile] Done. Dockerfile written at: {dockerfile_path}")
    return dockerfile_path
