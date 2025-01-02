# agnet_gpt.py
from agent_gpt.trainer import AgentGPTTrainer
from agent_gpt.predictor import AgentGPTPredictor
from agent_gpt.sagemaker_config import SageMakerConfig
from agent_gpt.hyperparams import Hyperparameters

class AgentGPT:
    def __init__(self):
        pass
    
    @classmethod
    def train(cls, sagemaker_config: SageMakerConfig, hyperparameters: Hyperparameters, env_simulator, **kwargs):
        if env_simulator == 'unity':
            from env_wrappers.unity_env import UnityEnv        # Interface for Unity environments
            env_simulator_cls = UnityEnv
        elif env_simulator == 'gym':
            from env_wrappers.gym_env import GymEnv            # Interface for Gym environments
            env_simulator_cls = GymEnv
        trainer: AgentGPTTrainer = AgentGPTTrainer(env_simulator_cls, hyperparameters.env_id, hyperparameters.env_url, **kwargs)
        trainer.sagemaker_train(sagemaker_config, hyperparameters)
        return trainer
        
    @classmethod
    def run(cls, sagemaker_config: SageMakerConfig, **kwargs):
        server = AgentGPTPredictor(sagemaker_config, **kwargs)
        server.sagemaker_run()
        return server 
