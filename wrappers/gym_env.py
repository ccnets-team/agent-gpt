# env_wrapper/gym_env.py
import gymnasium as gym

GymEnv = gym.Env

def get_gymnasium_envs(categories=None):
    """
    Retrieves environment IDs grouped by specified categories 
    based on entry points in the Gymnasium registry.
    
    Returns:
        list of str: All environment IDs in the specified categories.
    """
    from gymnasium import envs
    categories = categories or ["classic_control", "box2d", "toy_text", "mujoco", "phys2d", "tabular"]
    envs_by_category = {category: [] for category in categories}
    
    for env_spec in envs.registry.values():
        if isinstance(env_spec.entry_point, str):
            for category in categories:
                if category in env_spec.entry_point:
                    envs_by_category[category].append(env_spec.id)
                    break

    # Flatten all categories into a single list
    all_registered_envs = [env_id for env_list in envs_by_category.values() for env_id in env_list]
    return all_registered_envs