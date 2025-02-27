from dataclasses import dataclass, field, asdict
from typing import Optional, List

@dataclass
class ContainerConfig:
    image_name: str = ""
    deployment_name: str = ""
    container_ports: List[int] = field(default_factory=lambda: [80])
    
    env: str = "gym"               # Environment simulator: 'gym', 'unity', or 'custom'
    env_path: str = ""             # Path to the environment file/directory
    env_id: Optional[str] = None   # Optional environment identifier
    entry_point: Optional[str] = None  # Optional entry point for the environment
    additional_dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Returns a deep dictionary of all dataclass fields,
        including nested dataclasses.
        """
        return asdict(self)

    def set_config(self, **kwargs) -> None:
        """
        Updates the configuration with provided keyword arguments.
        Only existing attributes are updated.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
