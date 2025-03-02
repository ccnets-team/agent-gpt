from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict
from .network import get_network_info

@dataclass
class ContainerDeploymentConfig:
    deployment_name: str = ""
    image_name: str = ""
    additional_dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.deployment_name:
            self.deployment_name = "cloud-env-k8s"
        if not self.image_name:
            self.image_name = "cloud-env-k8s:latest"

@dataclass
class SimulatorConfig:
    env_type: str = "gym"               # Environment simulator: 'gym', 'unity', or 'custom'
    hosting: str = "cloud"       # Host type: 'local' or 'cloud'
    url: str = ""
    env_dir: str = None  # Path to the environment files directory
    available_ports: List[int] = field(default_factory=lambda: [34560, 34561, 34562, 34563])  # Local simulation ports
    container: ContainerDeploymentConfig = field(default_factory=ContainerDeploymentConfig)

@dataclass
class SimulatorRegistry:
    # A mapping from simulator identifier to its corresponding SimulatorConfig.
    simulators: Dict[str, SimulatorConfig] = field(default_factory=dict)
        
    def __post_init__(self):
        public_ip = get_network_info()['public_ip']
        if "local" not in self.simulators:
            self.simulators["local"] = SimulatorConfig(
                hosting="local", 
                url="http://" + public_ip,  
                env_type="gym"
            )
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]  # Adjust the index as needed
            self.simulators["local"].env_dir = str(project_root)
            
            
    def set_dockerfile(self, simulator_id: str) -> None:
        if simulator_id in self.simulators:
            from ..utils.deployment import create_dockerfile as _create_dockerfile
            env_type = self.simulators[simulator_id].env_type
            env_dir = self.simulators[simulator_id].env_dir
            additional_dependencies = self.simulators[simulator_id].container.additional_dependencies
            _create_dockerfile(env_type, env_dir, additional_dependencies)
        else:
            print(f"Warning: No simulator config found for identifier '{simulator_id}'")
    
    def set_k8s_manifest(self, simulator_id: str) -> None:
        if simulator_id in self.simulators:
            from ..utils.deployment import create_k8s_manifest as _create_k8s_manifest
            simulator = self.simulators[simulator_id]
            env_dir = simulator.env_dir
            available_ports = simulator.available_ports
            image_name = simulator.container.image_name
            deployment_name = simulator.container.deployment_name
            _create_k8s_manifest(env_dir, available_ports, image_name, deployment_name)
            
    def set_simulator(self, simulator_id: str, env_type: str = "gym", hosting: str = "cloud", url: str = None) -> None:
        valid_hosting_types = ["cloud", "remote", "local"]
        
        if simulator_id in self.simulators:
            print(f"Warning: Simulator config already exists for identifier '{simulator_id}'")
            return
        
        if hosting not in valid_hosting_types:
            print(f"Warning: host_type must be one of {valid_hosting_types}. Given: {hosting}")
            return
        
        self.simulators[simulator_id] = SimulatorConfig(
            env_type=env_type,
            hosting=hosting, 
            url=url, 
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
