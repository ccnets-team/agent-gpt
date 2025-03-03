from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        print("\n****************************************")
        print("Installation complete!")
        print("Note: The CLI command is 'agent-gpt' not 'agent-gpt-aws'.")
        print("Run 'agent-gpt --help' to see available commands and usage.")
        print("****************************************\n")

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    env_requirements = f.read().splitlines()

# Additional dependencies not in the file
cli_dependencies = [
    "typer>=0.15.1",
    "pyyaml>=6.0",
    "requests>=2.32.3",
    "boto3>=1.36.5",
    "sagemaker>=2.237.3",
    "kubernetes",
    "pyngrok",
]

# Combine the two lists
install_requires = env_requirements + cli_dependencies 

setup(
    name="agent-gpt-aws",
    version="0.3.7",
    packages=find_packages(), 
    entry_points={
        "console_scripts": [
            "agent-gpt=agent_gpt.cli:app",
            "agent-gpt-aws=agent_gpt.cli_deprecated:main",
        ],
    },
    install_requires=install_requires,
    extras_require={
         "unity": ["mlagents_envs==0.30.0", "protobuf==3.20.0"],
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
    python_requires=">=3.7",
    cmdclass={  
         'install': PostInstallCommand,
    },
)
