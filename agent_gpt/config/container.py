from dataclasses import dataclass, field, asdict
from typing import Optional, List

@dataclass
class DockerfileConfig:
    env: str = "gym"               # Environment simulator: 'gym', 'unity', or 'custom'
    env_id: Optional[str] = None   # Optional environment identifier
    entry_point: Optional[str] = None  # Optional entry point for the environment
    additional_dependencies: List[str] = field(default_factory=list)
        
@dataclass
class K8SManifestConfig:
    image_name: str = ""
    deployment_name: str = ""
    container_ports: List[int] = field(default_factory=list)
    def __post_init__(self):
        if not self.container_ports:
            self.container_ports = [80]

        if not self.deployment_name:
            self.deployment_name = "agent-gpt-cloud-env-k8s"                 

@dataclass
class ContainerConfig:
    env_path: str = ""             # Path to the environment file
    dockerfile: DockerfileConfig = field(default_factory=DockerfileConfig)
    k8s_manifest: K8SManifestConfig = field(default_factory=K8SManifestConfig)

    def __post_init__(self):
        # Convert nested dictionaries to their respective dataclass instances if needed.
        if isinstance(self.dockerfile, dict):
            self.dockerfile = DockerfileConfig(**self.dockerfile)
        if isinstance(self.k8s_manifest, dict):
            self.k8s_manifest = K8SManifestConfig(**self.k8s_manifest)
    
    def to_dict(self) -> dict:
        """
        Returns a deep dictionary of all dataclass fields,
        including nested dataclasses.
        """
        return asdict(self)

    def set_config(self, **kwargs) -> None:
        """
        Update the ContainerConfig instance using provided keyword arguments.
        For nested fields like 'dockerfile' and 'k8s_manifest', update only the specified sub-attributes.
        """
        for k, v in kwargs.items():
            if k == "dockerfile" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.dockerfile, sub_key):
                        setattr(self.dockerfile, sub_key, sub_value)
                    else:
                        print(f"Warning: dockerfile has no attribute '{sub_key}'")
            elif k == "k8s_manifest" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.k8s_manifest, sub_key):
                        setattr(self.k8s_manifest, sub_key, sub_value)
                    else:
                        print(f"Warning: k8s_manifest has no attribute '{sub_key}'")
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in ContainerConfig")
    
    def create_dockerfile(self):
        # Import locally to avoid circular dependency
        from ..utils.deployment import create_dockerfile as _create_dockerfile
        _create_dockerfile(self.env_path, self.dockerfile)

    def create_k8s_manifest(self):
        # Import locally to avoid circular dependency
        from ..utils.deployment import create_k8s_manifest as _create_k8s_manifest
        _create_k8s_manifest(self.env_path, self.k8s_manifest)
    
    def compose_container(self):
        self.create_dockerfile()
        self.create_k8s_manifest()