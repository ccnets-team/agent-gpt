import os
import yaml

from ..config.simulator import SimulatorRegistry
from ..config.network import NetworkConfig
from ..config.hyperparams import Hyperparameters
from ..config.sagemaker import SageMakerConfig

try:    
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    CURRENT_AGENT_GPT_VERSION = version("agent-gpt-aws")  # Replace with your package name
except PackageNotFoundError:
    CURRENT_AGENT_GPT_VERSION = "unknown"  # Fallback if the package is not installed

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.agent_gpt/config.yaml")

TOP_CONFIG_CLASS_MAP = {
    "simulator_registry": SimulatorRegistry,
    "network": NetworkConfig,
    "hyperparams": Hyperparameters,
    "sagemaker": SageMakerConfig,
}

def load_config() -> dict:
    if os.path.exists(DEFAULT_CONFIG_PATH):
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        # If the stored version doesn't match, clear the config file
        if config.get("version") != CURRENT_AGENT_GPT_VERSION:
            os.remove(DEFAULT_CONFIG_PATH)
            config = {}  # Start with an empty config
        return config
    return {}

def save_config(config_data: dict) -> None:
    config_data["version"] = CURRENT_AGENT_GPT_VERSION
    os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
    with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

def generate_section_config(section: str) -> dict:
    cls = TOP_CONFIG_CLASS_MAP.get(section)
    if cls:
        # With __post_init__ in NetworkConfig, simply instantiating is enough.
        return cls().to_dict()
    return {}

def initialize_config() -> dict:
    return { section: generate_section_config(section) for section in TOP_CONFIG_CLASS_MAP }

def convert_to_objects(overrides: dict) -> dict:
    """
    Instantiate top-level configuration objects and apply stored overrides.
    """
    result = {}
    for key, cls in TOP_CONFIG_CLASS_MAP.items():
        obj = cls()  # __post_init__ in NetworkConfig will fetch network info automatically.
        obj.set_config(**overrides.get(key, {}))
        result[key] = obj
    return result

def parse_value(value: str):
    """
    Try converting the string to int, float, or bool.
    If all conversions fail, return the string.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    if value is not None:
        lower = value.lower()
        if lower in ["true", "false"]:
            return lower == "true"
    return value

def parse_extra_args(args: list[str]) -> dict:
    """
    Parses extra CLI arguments provided in the form:
      --key value [value ...]
    Supports nested keys via dot notation, e.g.:
      --env_hosts.local1.env_endpoint "http://example.com:8500"
    Returns a nested dictionary of the parsed values.
    """
    new_changes = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]  # remove the leading "--"
            i += 1
            # Gather all subsequent arguments that do not start with '--'
            values = []
            while i < len(args) and not args[i].startswith("--"):
                values.append(args[i])
                i += 1

            # Determine if we have no values, a single value, or multiple values.
            if not values:
                parsed_value = None
            elif len(values) == 1:
                parsed_value = parse_value(values[0])
            else:
                parsed_value = [parse_value(val) for val in values]

            # Build a nested dictionary using dot notation.
            keys = key.split(".")
            d = new_changes
            for sub_key in keys[:-1]:
                d = d.setdefault(sub_key, {})
            d[keys[-1]] = parsed_value
        else:
            i += 1
    return new_changes

def recursive_update(target, changes: dict, prefix="") -> tuple:
    """
    Recursively update attributes of an object (or dictionary) using a nested changes dict.
    Only updates existing attributes/keys.

    Returns:
        tuple: (changed, diffs)
            changed (bool): True if any update was made, False otherwise.
            diffs (list): A list of differences in the form (full_key, old_value, new_value)
    """
    changed = False
    diffs = []

    if isinstance(target, dict):
        for k, v in changes.items():
            if k in target:
                current_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    sub_changed, sub_diffs = recursive_update(target[k], v, prefix=current_key)
                    if sub_changed:
                        changed = True
                        diffs.extend(sub_diffs)
                else:
                    if target[k] != v:
                        old_val = target[k]
                        target[k] = v
                        changed = True
                        diffs.append((current_key, old_val, v))
            # Do not add new keys.
    else:
        for attr, new_val in changes.items():
            if not hasattr(target, attr):
                continue
            current_val = getattr(target, attr)
            current_key = f"{prefix}.{attr}" if prefix else attr
            # If the new value is a dict, try to update the inner attributes.
            if isinstance(new_val, dict):
                sub_changed, sub_diffs = recursive_update(current_val, new_val, prefix=current_key)
                if sub_changed:
                    changed = True
                    diffs.extend(sub_diffs)
            else:
                if current_val != new_val:
                    old_val = current_val
                    setattr(target, attr, new_val)
                    changed = True
                    diffs.append((current_key, old_val, new_val))

    return changed, diffs

def apply_config_updates(config_obj, new_changes):
    # List to collect change summaries.
    list_changes = []
    # Loop through the parsed changes.
    for key, value in new_changes.items():
        diffs_for_key = []
        changed = False
        
        # Allow shorthand syntax for top-level config sections 
        # (simulator_registry, network, hyperparams, sagemaker).
        if key in (config_obj.keys()) and isinstance(value, dict) and len(value) == 1:
            inner_key, inner_value = list(value.items())[0]
            key = inner_key
            value = inner_value
            
        # Otherwise, update all config objects that have the attribute.
        for obj in config_obj.values():
            if not hasattr(obj, key):
                continue
            attr = getattr(obj, key)
            if callable(attr):
                if not isinstance(value, list):
                    value = [value]
                # Filter out None values if necessary.
                converted_args = [parse_value(arg) for arg in value if arg is not None]
                if converted_args:
                    attr(*converted_args)
                else:
                    attr()
                arg_str = " ".join(str(x) for x in converted_args)
                diffs_for_key.append((key, None, arg_str))
                changed = True
            elif isinstance(value, dict):
                ch, diffs = recursive_update(attr, value, prefix=key)
                if ch:
                    changed = True
                    diffs_for_key.extend(diffs)
            else:
                current_val = getattr(obj, key)
                if current_val != value:
                    setattr(obj, key, value)
                    changed = True
                    diffs_for_key.append((key, current_val, value))
        list_changes.append((key, value, changed, diffs_for_key))
    return list_changes