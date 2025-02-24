from setuptools import setup, find_packages

setup(
    name="agent-gpt",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "typer",
        "pyyaml",
        "boto3",
        "sagemaker",
        "gymnasium",
        # Add other dependencies as needed...
    ],
    entry_points={
        "console_scripts": [
            "agent-gpt=agent_gpt_cli:app",
        ],
    },
    author="Your Name",
    description="AgentGPT CLI for training and inference on AWS SageMaker",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
