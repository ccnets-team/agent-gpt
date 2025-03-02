from pathlib import Path                        
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict
from .network import get_network_info

@dataclass
class ContainerDeploymentConfig:
    deployment_name: str = "cloud-env-k8s"
    image_uri: str = None
    additional_dependencies: List[str] = field(default_factory=list)
 
@dataclass
class SimulatorConfig:
    env_type: str = "gym"               # Environment simulator: 'gym', 'unity', or 'custom'
    hosting: str = "cloud"       # Host type: 'local' or 'cloud'
    url: str = ""
    host: str = "0.0.0.0"
    connection: str = "tunnel"  # local: ip, tunnel(ngrok); cloud(aws): ec2, eks, app_runner
    total_agents: int = 128
    env_dir: str = None  # Path to the environment files directory
    ports: List[int] = field(default_factory=lambda: [34560, 34561, 34562, 34563])  # Local simulation ports
    container: ContainerDeploymentConfig = field(default_factory=ContainerDeploymentConfig)
        
    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if k == "container" and isinstance(v, dict):
                for sub_key, sub_value in v.items():
                    if hasattr(self.container, sub_key):
                        setattr(self.container, sub_key, sub_value)
                    else:
                        print(f"Warning: SimulatorConfig has no attribute '{sub_key}'")
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: No attribute '{k}' in SimulatorConfig")

    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass
class SimulatorRegistry:
    # A mapping from simulator identifier to its corresponding SimulatorConfig.
    simulators: Dict[str, SimulatorConfig] = field(default_factory=dict)
        
    def __post_init__(self):
        # Local simulator for direct connection
        if "local" not in self.simulators:
            network_info = get_network_info()
            ip = network_info['public_ip']
            self.simulators["local"] = SimulatorConfig(
                hosting="local", 
                url="http://" + ip,  
                env_type="gym",
                connection="tunnel",
            )
            project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
            self.simulators["local"].env_dir = str(project_root)
            self.simulators["local"].container.deployment_name = None
            
    # set dockerfile will be renamed to upload to cloud 
    # and the command will be named as "upload"
    def set_dockerfile(self, simulator_id: str) -> None:
        if simulator_id in self.simulators:
            from ..utils.deployment import create_dockerfile
            env_type = self.simulators[simulator_id].env_type
            env_dir = self.simulators[simulator_id].env_dir
            additional_dependencies = self.simulators[simulator_id].container.additional_dependencies
            create_dockerfile(env_type, env_dir, additional_dependencies)
        else:
            print(f"Warning: No simulator config found for identifier '{simulator_id}'")
    
    # command will be renamed to "simulate"
    def simulate_on_cloud(self, simulator_id: str) -> None:
        if simulator_id in self.simulators:
            from ..utils.deployment import deploy_eks_simulator, service_eks_simulator
            simulator = self.simulators[simulator_id]
            ports = simulator.ports
            image_uri = simulator.container.image_uri
            deployment_name = simulator.container.deployment_name
            deploy_eks_simulator(deployment_name, image_uri, ports)
            service_eks_simulator(deployment_name, ports)
        else:
            print(f"Warning: No simulator config found for identifier '{simulator_id}'")
            
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
