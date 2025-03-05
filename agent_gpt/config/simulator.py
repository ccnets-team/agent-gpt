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
    hosting: str = "local"       # Host type: 'local' or 'cloud'
    connection: str = "tunnel"  # local: ip, tunnel(ngrok); cloud(aws): ec2, eks, app_runner
    url: str = ""
    host: str = "0.0.0.0"
    total_agents: int = 128
    env_dir: str = None  # Path to the environment files directory
    ports: List[int] = field(default_factory=lambda: [34560, 34561])  # Local simulation ports
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

    def to_dict(self) -> Dict:
        return asdict(self)
    
@dataclass
class SimulatorRegistry:
    # A mapping from simulator identifier to its corresponding SimulatorConfig.
    simulator: Dict[str, SimulatorConfig] = field(default_factory=dict)
        
    def __post_init__(self):
        # Local simulator for direct connection
        if "local-tunnel" not in self.simulator:
            network_info = get_network_info()
            ip = network_info['public_ip']
            self.simulator["local-tunnel"] = SimulatorConfig(
                hosting="local", 
                url="http://" + ip,  
                env_type="gym",
                connection="tunnel",
                ports = [45670, 45671],
            )
            project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
            self.simulator["local-tunnel"].env_dir = str(project_root)
            self.simulator["local-tunnel"].container.deployment_name = None

        if "local-ip" not in self.simulator:
            network_info = get_network_info()
            ip = network_info['public_ip']
            self.simulator["local-ip"] = SimulatorConfig(
                hosting="local", 
                url="http://" + ip,  
                env_type="gym",
                connection="ip",
                ports = [34560, 34561],
            )
            project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
            self.simulator["local-ip"].env_dir = str(project_root)
            self.simulator["local-ip"].container.deployment_name = None

            
    # set dockerfile will be renamed to upload to cloud 
    # and the command will be named as "upload"
    def set_dockerfile(self, simulator_id: str) -> None:
        if simulator_id in self.simulator:
            from ..utils.deployment import create_dockerfile
            env_type = self.simulator[simulator_id].env_type
            env_dir = self.simulator[simulator_id].env_dir
            additional_dependencies = self.simulator[simulator_id].container.additional_dependencies
            create_dockerfile(env_type, env_dir, additional_dependencies)
        else:
            print(f"Warning: No simulator config found for identifier '{simulator_id}'")
    
    # command will be renamed to "simulate"
    def simulate_on_cloud(self, simulator_id: str) -> None:
        if simulator_id in self.simulator:
            from ..utils.deployment import deploy_eks_simulator, service_eks_simulator
            simulator = self.simulator[simulator_id]
            ports = simulator.ports
            image_uri = simulator.container.image_uri
            deployment_name = simulator.container.deployment_name
            deploy_eks_simulator(deployment_name, image_uri, ports)
            service_eks_simulator(deployment_name, ports)
        else:
            print(f"Warning: No simulator config found for identifier '{simulator_id}'")
            
    def set_simulator(
        self, 
        simulator_id: str, 
        env_type: str = None, 
        hosting: str = None, 
        connection: str = None,
        url: str = '',
        host: str = "0.0.0.0", 
        total_agents: int = 128, 
        env_dir: str = None, 
        ports: List[int] = None
    ) -> None:

        if simulator_id in self.simulator:
            raise ValueError(f"Simulator identifier '{simulator_id}' already exists. Use a different identifier.")

        if hosting:
            valid_hosting_types = ["local", "cloud"]
            if hosting not in valid_hosting_types:
                raise ValueError(f"host_type must be one of {valid_hosting_types}. Given: {hosting}")
            if hosting == "local" and connection not in ["ip", "tunnel"]:
                raise ValueError(f"Connection type must be one of ['ip', 'tunnel'] for local hosting. Given: {connection}")
            if hosting == "cloud" and connection not in ["ec2", "eks", "app_runner"]:
                raise ValueError(f"Connection type must be one of ['ec2', 'eks', 'app_runner'] for cloud hosting. Given: {connection}")

        if env_type not in ["gym", "unity"]:
            print(f"Make sure the environment type is not one of ['gym', 'unity'] unless using custom env: {env_type}")

        self.simulator[simulator_id] = SimulatorConfig(
            env_type=env_type,
            hosting=hosting, 
            connection=connection,
            url=url,
            host=host,
            total_agents=total_agents,
            env_dir=env_dir,
            ports=ports
        )
        if hosting == "local" and not url:
            network_info = get_network_info()
            ip = network_info['public_ip']
            self.simulator[simulator_id].url = "http://" + ip
    
    def del_simulator(self, simulator_id: str) -> None:
        if simulator_id in self.simulator:
            del self.simulator[simulator_id]
        else:
            raise ValueError(f"Simulator identifier '{simulator_id}' not found.")
    
    def to_dict(self) -> Dict:
        return asdict(self)
        
    def set_config(self, **kwargs) -> None:
        """
        Update nested simulator configurations.
        Expects a key "simulator" in kwargs whose value is a dict mapping simulator
        identifiers to their updates.
        """
        def update_dataclass(instance, updates: Dict):
            for key, value in updates.items():
                if hasattr(instance, key):
                    attr = getattr(instance, key)
                    if is_dataclass(attr) and isinstance(value, dict):
                        update_dataclass(attr, value)
                    else:
                        setattr(instance, key, value)
                else:
                    print(f"Warning: {instance.__class__.__name__} has no attribute '{key}'")
        
        simulators_data = kwargs.get("simulator", {})
        for simulator_id, simulator_updates in simulators_data.items():
            if simulator_id not in self.simulator:
                self.simulator[simulator_id] = SimulatorConfig()  # all defaults applied
            update_dataclass(self.simulator[simulator_id], simulator_updates)
