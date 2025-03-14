from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    env_requirements = f.read().splitlines()

# Additional dependencies not in the file
cli_dependencies = [
    "typer",
    "websocket-client",
    "pyyaml",
    "boto3",
    "sagemaker",
]

# Combine the two lists
install_requires = env_requirements + cli_dependencies 

setup(
    name="agent-gpt-aws",
    version="0.9.3",
    packages=find_packages(), 
    include_package_data=True,
    package_data={
        "agent_gpt": ["*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "agent-gpt=agent_gpt.cli:app",
            "agent-gpt-aws=agent_gpt.cli_deprecated:main",
        ],
    },
    install_requires=install_requires,
    extras_require={
         "mujoco": ["gymnasium[mujoco]"],
         "mlagents": ["mlagents_envs==0.30.0", "protobuf==3.20.0"],
    },        
    author="JunHo Park",
    author_email="junho@ccnets.org",
    url="https://github.com/ccnets-team/agent-gpt",
    description="AgentGPT CLI for training and inference on AWS SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Dual Licensed (AGENT GPT COMMERCIAL LICENSE or GNU GPLv3)",
    keywords="agent gpt reinforcement-learning sagemaker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
