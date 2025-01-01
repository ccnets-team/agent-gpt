# agnet_gpt.py
from remote.agent_gpt_predictor import AgentGPTPredictor
from remote.agent_gpt_trainer import AgentGPTTrainer
from .gpt_trainer_config import Hyperparameters, SageMakerConfig

class AgentGPT:
    def __init__(self):
        pass
    
    @classmethod
    def run(cls, sagemaker_config: SageMakerConfig, **kwargs):
        server = AgentGPTPredictor(sagemaker_config, **kwargs)
        return server.sagemaker_run()
    
    @classmethod
    def train(cls, sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters, env_simulator, **kwargs):
        if env_simulator == 'unity':
            from envs.unity_env import UnityEnv        # Interface for Unity environments
            env_simulator_cls = UnityEnv
        elif env_simulator == 'gym':
            from envs.gym_env import GymEnv            # Interface for Gym environments
            env_simulator_cls = GymEnv
        trainer: AgentGPTTrainer = AgentGPTTrainer(env_simulator_cls, hyperparameters.env_id, **kwargs)
        trainer.sagemaker_train(sagemaker_config, hyperparameters)