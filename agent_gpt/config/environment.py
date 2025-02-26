from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional, List, Dict
from .network import get_network_info

@dataclass
class DockerfileConfig:
    additional_dependencies: List[str] = field(default_factory=list)
       
@dataclass
class K8SManifestConfig:
    image_name: str = ""
    deployment_name: str = ""
    # Removed container_ports so that local ports are defined at the environment level.
    
    def __post_init__(self):
        if not self.deployment_name:
            self.deployment_name = "agent-gpt-cloud-env-k8s"

@dataclass
class EnvironmentConfig:
    env: str = "gym"               # Environment simulator: 'gym', 'unity', or 'custom'
    host_type: str = "cloud"       # Host type: 'local' or 'cloud'
    host_name: str = ""
    env_id: Optional[str] = None   # Optional environment identifier
    entry_point: Optional[str] = None
    ports: List[int] = field(default_factory=lambda: [80])  # Local simulation ports

    env_path: str = ""             # Path to the environment file for docker build
    dockerfile: DockerfileConfig = field(default_factory=DockerfileConfig)
    k8s_manifest: K8SManifestConfig = field(default_factory=K8SManifestConfig)
    
    # (You may include create_dockerfile/manifest methods as needed.)
    def create_dockerfile(self):
        from ..utils.deployment import create_dockerfile as _create_dockerfile
        _create_dockerfile(self.env_path, self.env, self.env_id, self.entry_point, self.dockerfile)

    def create_k8s_manifest(self):
        from ..utils.deployment import create_k8s_manifest as _create_k8s_manifest
        _create_k8s_manifest(self.env_path, self.ports, self.k8s_manifest)

@dataclass
class MultiEnvironmentConfig:
    # A mapping from environment identifier to its corresponding EnvironmentConfig.
    envs: Dict[str, EnvironmentConfig] = field(default_factory=dict)
        
    def __post_init__(self):
        public_ip = get_network_info()['public_ip']
        if "local" not in self.envs:
            self.envs["local"] = EnvironmentConfig(
                host_type="local", 
                host_name="http://" + public_ip,  
                env="gym"
            )
    
    def set_env(self, env_identifier: str, host_type: str = "local", host_name: str = None, env: str = "gym") -> None:
        valid_host_types = ["cloud", "remote", "local"]
        
        if env_identifier in self.envs:
            print(f"Warning: Environment config already exists for identifier '{env_identifier}'")
            return
        if host_type not in valid_host_types:
            print(f"Warning: host_type must be one of {valid_host_types}. Given: {host_type}")
            return
        self.envs[env_identifier] = EnvironmentConfig(
            host_type=host_type, 
            host_name=host_name, 
            env=env
        )
    
    def del_env(self, env_identifier: str) -> None:
        if env_identifier in self.envs:
            del self.envs[env_identifier]
        else:
            print(f"Warning: No environment config found for identifier '{env_identifier}'")
    
    def to_dict(self) -> dict:
        # Return a flat dictionary of env configs.
        return asdict(self)
        
    def set_config(self, **kwargs) -> None:
        """
        Update nested environment configurations.
        Expects a key "envs" in kwargs whose value is a dict mapping environment
        identifiers to their updates.
        """
        def update_dataclass(instance, updates: dict):
            for key, value in updates.items():
                if hasattr(instance, key):
                    attr = getattr(instance, key)
                    if is_dataclass(attr) and isinstance(value, dict):
                        update_dataclass(attr, value)
                    else:
                        setattr(instance, key, value)
                else:
                    print(f"Warning: {instance.__class__.__name__} has no attribute '{key}'")
        
        envs_data = kwargs.get("envs", {})
        for env_identifier, env_updates in envs_data.items():
            # If the environment doesn't exist yet, create a new default instance.
            if env_identifier not in self.envs:
                self.envs[env_identifier] = EnvironmentConfig()  # all defaults applied
                print(f"Created new environment config for identifier '{env_identifier}'")
            env_config = self.envs.get(env_identifier)
            update_dataclass(env_config, env_updates)                
