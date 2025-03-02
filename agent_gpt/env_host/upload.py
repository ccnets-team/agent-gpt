import os
import base64
import logging
import docker
import boto3
from ..config.simulator import SimulatorConfig

logger = logging.getLogger(__name__)
TAG = "latest"

def upload_simulator(region, simulator_config: SimulatorConfig) -> None:
    from ..utils.deployment import create_dockerfile

    # Generate Dockerfile for the simulator.
    env_type = simulator_config.env_type
    env_dir = simulator_config.env_dir
    additional_dependencies = simulator_config.container.additional_dependencies
    dockerfile_path = create_dockerfile(env_type, env_dir, additional_dependencies)
    
    logger.info(f"Dockerfile generated at: {dockerfile_path}")
    
    repository_name = simulator_config.container.deployment_name.lower()
    
    docker_client = docker.from_env()
    local_tag = f"{repository_name}:{TAG}"
    logger.info(f"Building Docker image locally as: {local_tag}")
    try:
        build_context = os.path.dirname(dockerfile_path)
        image, build_logs = docker_client.images.build(path=build_context, tag=local_tag)
        for chunk in build_logs:
            if 'stream' in chunk:
                logger.info(chunk['stream'].strip())
        logger.info(f"Local Docker image built successfully: {local_tag}")
    except Exception as e:
        logger.error("Docker image build failed:", exc_info=e)
        raise

    # Process the specified region.
    logger.info("=" * 40)
    logger.info(f"Processing region: {region}")

    ecr = boto3.client('ecr', region_name=region)
    try:
        ecr.describe_repositories(repositoryNames=[repository_name])
        logger.info(f"[{region}] Repository {repository_name} already exists.")
    except ecr.exceptions.RepositoryNotFoundException:
        logger.info(f"[{region}] Repository {repository_name} not found. Creating it...")
        ecr.create_repository(repositoryName=repository_name)
        logger.info(f"[{region}] Repository {repository_name} created.")

    try:
        auth_response = ecr.get_authorization_token()
        auth_data = auth_response['authorizationData'][0]
        token = auth_data['authorizationToken']
        decoded = base64.b64decode(token).decode('utf-8')
        username, password = decoded.split(':')
        registry = auth_data['proxyEndpoint'].replace("https://", "")
        docker_client.login(username=username, password=password, registry=registry)
        logger.info(f"[{region}] Docker login succeeded for registry: {registry}")
    except Exception as e:
        logger.error(f"[{region}] Docker login failed:", exc_info=e)
        raise

    target_image = f"{registry}/{repository_name}:{TAG}"
    logger.info(f"[{region}] Tagging image as: {target_image}")
    if not image.tag(target_image):
        logger.error(f"[{region}] Failed to tag image with {target_image}")
        raise Exception("Image tagging failed")

    logger.info(f"[{region}] Pushing image to: {target_image}")
    try:
        push_logs = docker_client.images.push(target_image, stream=True, decode=True)
        for log in push_logs:
            logger.info(log)
        logger.info(f"[{region}] Image pushed successfully to: {target_image}")
    except Exception as e:
        logger.error(f"[{region}] Image push failed:", exc_info=e)
        raise

    try:
        response = ecr.batch_get_image(
            repositoryName=repository_name,
            imageIds=[{'imageTag': TAG}],
            acceptedMediaTypes=['application/vnd.docker.distribution.manifest.v2+json']
        )
        images = response.get('images', [])
        if images:
            manifest = images[0]['imageManifest']
            logger.info(f"[{region}] Docker manifest for image {target_image}:\n{manifest}")
        else:
            logger.info(f"[{region}] No image manifest found for {repository_name}:{TAG}")
    except Exception as e:
        logger.error(f"[{region}] Docker manifest inspect failed:", exc_info=e)
        raise

    logger.info("=" * 40)

    # If we reach this point without any errors, all steps were successful.
    simulator_config.container.image_uri = target_image
    logger.info(f"Simulator config updated with image URI: {target_image}")
