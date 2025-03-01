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
class SimulatorConfig:
    host_type: str = "cloud"       # Host type: 'local' or 'cloud'
    host_name: str = ""
    env_type: str = "gym"               # Environment simulator: 'gym', 'unity', or 'custom'
    env_path: str = ""             # Path to the environment file for docker build
    
    ports: List[int] = field(default_factory=lambda: [80])  # Local simulation ports

    dockerfile: DockerfileConfig = field(default_factory=DockerfileConfig)
    k8s_manifest: K8SManifestConfig = field(default_factory=K8SManifestConfig)
    
    # (You may include create_dockerfile/manifest methods as needed.)
    def create_dockerfile(self):
        from ..utils.deployment import create_dockerfile as _create_dockerfile
        _create_dockerfile(self.env_path, self.env_type, self.dockerfile)

    def create_k8s_manifest(self):
        from ..utils.deployment import create_k8s_manifest as _create_k8s_manifest
        _create_k8s_manifest(self.env_path, self.ports, self.k8s_manifest)

@dataclass
class SimulatorRegistry:
    # A mapping from simulator identifier to its corresponding SimulatorConfig.
    simulators: Dict[str, SimulatorConfig] = field(default_factory=dict)
        
    def __post_init__(self):
        public_ip = get_network_info()['public_ip']
        if "local" not in self.simulators:
            self.simulators["local"] = SimulatorConfig(
                host_type="local", 
                host_name="http://" + public_ip,  
                env_type="gym"
            )
    
    def set_simulator(self, simulator_id: str, host_type: str = "local", host_name: str = None, env_type: str = "gym") -> None:
        valid_host_types = ["cloud", "remote", "local"]
        
        if simulator_id in self.simulators:
            print(f"Warning: Simulator config already exists for identifier '{simulator_id}'")
            return
        if host_type not in valid_host_types:
            print(f"Warning: host_type must be one of {valid_host_types}. Given: {host_type}")
            return
        self.simulators[simulator_id] = SimulatorConfig(
            host_type=host_type, 
            host_name=host_name, 
            env_type=env_type
        )
    
    def del_simulator(self, simulator_id: str) -> None:
        if simulator_id in self.simulators:
            del self.simulators[simulator_id]
        else:
            print(f"Warning: No simulator config found for identifier '{simulator_id}'")
    
    def to_dict(self) -> dict:
        return asdict(self)
        
    def set_config(self, **kwargs) -> None:
        """
        Update nested simulator configurations.
        Expects a key "simulators" in kwargs whose value is a dict mapping simulator
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
        
        simulators_data = kwargs.get("simulators", {})
        for simulator_id, simulator_updates in simulators_data.items():
            if simulator_id not in self.simulators:
                self.simulators[simulator_id] = SimulatorConfig()  # all defaults applied
                print(f"Created new simulator config for identifier '{simulator_id}'")
            simulator_config = self.simulators.get(simulator_id)
            update_dataclass(simulator_config, simulator_updates)
