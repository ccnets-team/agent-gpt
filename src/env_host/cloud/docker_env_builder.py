import os
import subprocess
import platform

def detect_docker_cmd():
    def _check_command_exists(command: str) -> bool:
        from shutil import which
        return which(command) is not None
                
    # Detect operating system and set docker_cmd accordingly
    os_name = platform.system().lower()
    if "windows" in os_name:
        # Attempt 'docker.exe' or fallback to 'docker'
        possible_docker_names = ["docker.exe", "docker"]
        for name in possible_docker_names:
            if _check_command_exists(name):
                docker_cmd = name
                break
        else:
            raise FileNotFoundError("Could not find 'docker.exe' or 'docker' in PATH on Windows.")
    else:
        if _check_command_exists("docker"):
            docker_cmd = "docker"
        else:
            raise FileNotFoundError("Could not find 'docker' in PATH on this OS.")
    return os_name, docker_cmd


def build_docker_image_impl(docker_cmd, local_image_name, dockerfile_path):
    """
    Builds a Docker image using the specified Dockerfile and tags it with `local_image_name`.
    Prints Docker's stdout/stderr for debugging if something goes wrong,
    then prints Docker client version if successful.
    """
    context_dir = os.path.dirname(dockerfile_path) or "."

    build_command = [
        docker_cmd,
        "build",
        "-t",
        local_image_name,
        "-f",
        dockerfile_path,
        context_dir
    ]

    print(f"[DockerEnvBuilder] Building Docker image '{local_image_name}' "
          f"from Dockerfile: {dockerfile_path} (context='{context_dir}')")

    process = subprocess.run(build_command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        print("[DEBUG] Docker build failed:")
        print("[DEBUG] STDOUT:", process.stdout)
        print("[DEBUG] STDERR:", process.stderr)
        print("[Hint] Make sure Docker is installed and running. Check Docker Desktop on Mac/Windows, or daemon on Linux.")
        raise subprocess.CalledProcessError(process.returncode, build_command, process.stdout, process.stderr)
    else:
        print(f"[DockerEnvBuilder] Successfully built image '{local_image_name}'")

    # Print Docker client version
    version_command = [docker_cmd, "version", "--format", "{{.Client.Version}}"]
    version_result = subprocess.run(version_command, capture_output=True, text=True)
    if version_result.returncode == 0:
        docker_version = version_result.stdout.strip()
        print(f"[DockerEnvBuilder] Docker client version: {docker_version}")
    else:
        print("[DockerEnvBuilder] Could not retrieve Docker version (non-zero return code).")


def tag_docker_image_impl(docker_cmd: str, local_image_name: str, new_image_name: str):
    """
    Tags a local Docker image with a new name or registry reference.
    Shows debugging info if the command fails.
    """
    print(f"[DockerEnvBuilder] Tagging '{local_image_name}' as '{new_image_name}'")

    tag_command = [
        docker_cmd,
        "tag",
        local_image_name,
        new_image_name
    ]

    process = subprocess.run(tag_command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        print("[DEBUG] Docker tag failed:")
        print("[DEBUG] STDOUT:", process.stdout)
        print("[DEBUG] STDERR:", process.stderr)
        print("[Hint] Make sure Docker is running and you have permission to tag images.")
        raise subprocess.CalledProcessError(process.returncode, tag_command, process.stdout, process.stderr)
    else:
        print(f"[DockerEnvBuilder] Successfully tagged image: {new_image_name}")

    # Print Docker client version
    version_command = [docker_cmd, "version", "--format", "{{.Client.Version}}"]
    version_result = subprocess.run(version_command, capture_output=True, text=True)
    if version_result.returncode == 0:
        docker_version = version_result.stdout.strip()
        print(f"[DockerEnvBuilder] Docker client version: {docker_version}")
    else:
        print("[DockerEnvBuilder] Could not retrieve Docker version (non-zero return code).")

def push_docker_image_impl(
    docker_cmd: str,
    ecr_registry: str,
    local_image_name: str,
    remote_image_name: str,
    region: str = None,
    ensure_ecr_login: bool = False,
    ensure_ecr_repo: bool = False
):
    """
    Pushes a local Docker image to a remote registry (e.g., ECR).
    If do_ecr_login=True, attempts to log in to ECR first.
    If ensure_ecr_repo=True, tries to confirm or create the ECR repository before pushing.

    Manual Equivalent Steps:
    -----------------------------------------------------------
    (Optional) ECR Repo Creation:
        aws ecr create-repository --region <REGION> --repository-name <REPO_NAME>

    (Optional) ECR Login:
        aws ecr get-login-password --region <REGION> | \
            docker login --username AWS --password-stdin <ECR_REGISTRY>

    1) docker tag <LOCAL_IMAGE> <ECR_REGISTRY>/<REMOTE_IMAGE>
       e.g. docker tag my-env-image 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-env-image

    2) docker push <ECR_REGISTRY>/<REMOTE_IMAGE>
       e.g. docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-env-image

    3) docker version --format "{{.Client.Version}}"
       # to confirm Docker client version
    -----------------------------------------------------------
    """

    full_remote_name = f"{ecr_registry}/{remote_image_name}"

    # (Optional) Create or verify the repository in ECR
    if ensure_ecr_repo and region:
        # Extract the repository name from remote_image_name,
        # ignoring any possible :tag suffix. e.g. "my-repo:latest" -> "my-repo"
        # If remote_image_name includes subpaths, e.g. "somefolder/my-repo:latest",
        # then ECR repository would be "somefolder/my-repo".
        repo_name = remote_image_name
        # Remove any ":tag" if present
        if ":" in repo_name:
            repo_name = repo_name.split(":")[0]

        print(f"[DockerEnvBuilder] Checking or creating ECR repo: {repo_name} in region={region}")
        # 1) describe-repositories
        desc_cmd = [
            "aws", "ecr", "describe-repositories",
            "--region", region,
            "--repository-names", repo_name
        ]
        print("[DockerEnvBuilder] Running:", " ".join(desc_cmd))
        desc_process = subprocess.run(
            desc_cmd, capture_output=True, text=True, check=False
        )

        if desc_process.returncode != 0:
            # If it's "RepositoryNotFoundException", we create the repository
            if "RepositoryNotFoundException" in desc_process.stderr:
                print(f"[DockerEnvBuilder] ECR repo '{repo_name}' not found. Creating it...")
                create_cmd = [
                    "aws", "ecr", "create-repository",
                    "--region", region,
                    "--repository-name", repo_name
                ]
                print("[DockerEnvBuilder] Running:", " ".join(create_cmd))
                create_process = subprocess.run(
                    create_cmd, capture_output=True, text=True, check=False
                )
                if create_process.returncode != 0:
                    print("[DEBUG] ECR create-repository failed:")
                    print("[DEBUG] STDOUT:", create_process.stdout)
                    print("[DEBUG] STDERR:", create_process.stderr)
                    print("[Hint] Ensure AWS CLI is installed, credentials are valid, and you have permissions to create ECR repos.")
                    raise subprocess.CalledProcessError(
                        create_process.returncode, create_cmd,
                        create_process.stdout, create_process.stderr
                    )
                else:
                    print(f"[DockerEnvBuilder] Successfully created ECR repository '{repo_name}'.")
            else:
                # Some other describe error
                print("[DEBUG] ECR describe-repositories failed:")
                print("[DEBUG] STDOUT:", desc_process.stdout)
                print("[DEBUG] STDERR:", desc_process.stderr)
                print("[Hint] Ensure AWS CLI is installed, credentials are valid, and Docker is running.")
                raise subprocess.CalledProcessError(
                    desc_process.returncode, desc_cmd,
                    desc_process.stdout, desc_process.stderr
                )
        else:
            print(f"[DockerEnvBuilder] ECR repo '{repo_name}' already exists.")

    # (Optional) Log into ECR if requested
    if ensure_ecr_login and region:
        print(f"[DockerEnvBuilder] Logging into ECR registry: {ecr_registry} (region={region})")
        full_login_command = (
            f"aws ecr get-login-password --region {region} | "
            f"{docker_cmd} login --username AWS --password-stdin {ecr_registry}"
        )
        print(f"[DockerEnvBuilder] Running: {full_login_command}")

        login_process = subprocess.run(full_login_command, shell=True, capture_output=True, text=True, check=False)
        if login_process.returncode != 0:
            print("[DEBUG] ECR login failed:")
            print("[DEBUG] STDOUT:", login_process.stdout)
            print("[DEBUG] STDERR:", login_process.stderr)
            print("[Hint] Ensure AWS CLI is installed, credentials are valid, and Docker is running.")
            raise subprocess.CalledProcessError(
                login_process.returncode, full_login_command,
                login_process.stdout, login_process.stderr
            )
        else:
            print("[DockerEnvBuilder] ECR login successful.")

    # 1) Tag the local image
    tag_command = [
        docker_cmd,
        "tag",
        local_image_name,
        full_remote_name
    ]
    print(f"[DockerEnvBuilder] Tagging '{local_image_name}' as '{full_remote_name}'")

    tag_process = subprocess.run(tag_command, capture_output=True, text=True, check=False)
    if tag_process.returncode != 0:
        print("[DEBUG] Docker tag failed:")
        print("[DEBUG] STDOUT:", tag_process.stdout)
        print("[DEBUG] STDERR:", tag_process.stderr)
        print("[Hint] Make sure Docker is running and you have permission to tag images.")
        raise subprocess.CalledProcessError(
            tag_process.returncode, tag_command,
            tag_process.stdout, tag_process.stderr
        )
    else:
        print(f"[DockerEnvBuilder] Successfully tagged image: {full_remote_name}")

    # 2) Push the tagged image
    push_command = [
        docker_cmd,
        "push",
        full_remote_name
    ]
    print(f"[DockerEnvBuilder] Pushing image to '{full_remote_name}'")

    push_process = subprocess.run(push_command, capture_output=True, text=True, check=False)
    if push_process.returncode != 0:
        print("[DEBUG] Docker push failed:")
        print("[DEBUG] STDOUT:", push_process.stdout)
        print("[DEBUG] STDERR:", push_process.stderr)
        print("[Hint] Make sure Docker is running and you're logged in to your registry (e.g., ECR).")
        raise subprocess.CalledProcessError(
            push_process.returncode, push_command,
            push_process.stdout, push_process.stderr
        )
    else:
        print(f"[DockerEnvBuilder] Successfully pushed image to '{full_remote_name}'")

    # (Optional) Print Docker client version
    version_command = [docker_cmd, "version", "--format", "{{.Client.Version}}"]
    version_result = subprocess.run(version_command, capture_output=True, text=True)
    if version_result.returncode == 0:
        docker_version = version_result.stdout.strip()
        print(f"[DockerEnvBuilder] Docker client version: {docker_version}")
    else:
        print("[DockerEnvBuilder] Could not retrieve Docker version (non-zero return code).")

    return full_remote_name
