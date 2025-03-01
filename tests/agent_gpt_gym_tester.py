# tests/agent_gpt_gym_tester.py

import argparse
import gymnasium as gym
from agent_gpt.core import AgentGPT
from agent_gpt.config.sagemaker import SageMakerConfig, InferenceConfig

DEFAULT_ENDPOINT_NAME = "agent_gpt_gym_tester"
DEFAULT_RENDER_MODE = None
DEFAULT_NUM_EPISODES = 10
DEFAULT_INSTANCE_TYPE = "ml.t2.medium"

def run_episodes(agent_gpt, env_name, render_mode=DEFAULT_RENDER_MODE, num_episodes: int = DEFAULT_NUM_EPISODES):
    """
    Runs a given number of episodes using the Gym environment and the
    AgentGPT API (via agent_gpt.select_action) for action selection.
    For each episode, it accumulates the rewards and prints the cumulative reward.
    
    :param agent_gpt: A GPTAPI client obtained from AgentGPT.run_on_cloud().
    :param env_name: The Gym environment name.
    :param render_mode: Optional; if provided, passed to gym.make (e.g., "human").
    :param num_episodes: Number of episodes to run.
    :return: A list of cumulative rewards per episode.
    """
    env = gym.make(env_name, render_mode=render_mode)
    episode_rewards = []
    
    for ep in range(num_episodes):
        # Reset the environment with a seed for reproducibility.
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        agent_id = f"agent_{ep}"

        while not done:
            # Select an action using the public API.
            action_list = agent_gpt.select_action(agent_ids=[agent_id], observations=[obs])
            action = action_list[0]
            # print(f"Selected action: {action}")
            # Step the environment.
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            print(f"Step reward: {reward}, total reward: {total_reward}")
            done = terminated or truncated
            
        print(f"[TEST] status:\n{agent_gpt.status()}")
        print(f"Episode {ep+1} cumulative reward: {total_reward}")
        episode_rewards.append(total_reward)
    
    return episode_rewards

def parse_arguments(parser):
    parser.add_argument("--role-arn", type=str,
                        default="arn:aws:iam::<your-account-id>:role/<YourSageMakerRole>",
                        help="The AWS IAM Role ARN for SageMaker.")
    parser.add_argument("--model-data", type=str,
                        default="s3://<your-bucket>/agent-gpt-trainer/output/model.tar.gz",
                        help="S3 path to the model data tarball.")
    parser.add_argument("--instance-type", type=str,
                        default=DEFAULT_INSTANCE_TYPE,
                        help="SageMaker instance type.")
    parser.add_argument("--endpoint-name", type=str,
                        default=DEFAULT_ENDPOINT_NAME,
                        help="Endpoint name to use or create.")
    parser.add_argument("--env-id", type=str,
                        default="Humanoid-v5",
                        help="Gym environment name.")
    parser.add_argument("--render-mode", type=str,
                        default=DEFAULT_RENDER_MODE,
                        help="Gym render mode (e.g., 'human').")
    parser.add_argument("--num-episodes", type=int,
                        default=DEFAULT_NUM_EPISODES,
                        help="Number of episodes to run.")
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(description="AgentGPT Gym Tester")
    args = parse_arguments(parser)
    inference_config = InferenceConfig(
        model_data=args.model_data,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name
    )
    sagemaker_config = SageMakerConfig(
        role_arn=args.role_arn,
        inference=inference_config,
    )
    
    # Deploy (or reuse) the endpoint using AgentGPT to obtain a GPTAPI client.
    agent_gpt = AgentGPT.infer(sagemaker_config)
    
    # Run episodes and print cumulative rewards.
    cumulative_rewards = run_episodes(agent_gpt, env_name=args.env_id, render_mode=args.render_mode, num_episodes=args.num_episodes)
    print("Cumulative rewards per episode:", cumulative_rewards)

if __name__ == "__main__":
    main()
