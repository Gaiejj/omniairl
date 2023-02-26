import os
import yaml
from typing import Dict, Any

def get_default_yaml(algorithm_name: str, env_name: str) -> Dict[str, Any]:
    """
    Load environment configuration for a given algorithm and environment.

    Args:
        algorithm_name (str): The name of the algorithm to use.
        env_name (str): The name of the environment to use.

    Returns:
        A dictionary containing the environment configuration for the specified algorithm and environment.
        If the algorithm or environment is not found, the default configuration is returned.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    env_cfgs_path = os.path.join(path, '..', 'configs', 'env_configs', f'{env_name}.yaml')
    with open(env_cfgs_path, "r") as yaml_file:
        env_config = yaml.safe_load(yaml_file)
    env_config = env_config.get(algorithm_name, env_config["default"])
    algo_cfgs_path = os.path.join(path, '..', 'configs', 'algo_configs', f'{algorithm_name}.yaml')
    with open(algo_cfgs_path, "r") as yaml_file:
        algo_config = yaml.safe_load(yaml_file)
    algo_config = algo_config.get(env_name, algo_config["default"])
    return env_config, algo_config
