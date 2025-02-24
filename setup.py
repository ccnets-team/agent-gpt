from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-gpt-aws",  # Package name on PyPI
    version="0.1.0",
    packages=find_packages(where="agent_gpt"),
    package_dir={"": "agent_gpt"},
    entry_points={
        "console_scripts": [
            "agent-gpt=agent_gpt.cli:app",  # Reference the CLI inside the package
        ],
    },
    install_requires=[
        "typer",
        "pyyaml",
        "boto3",
        "uvicorn",
        "fastapi",
        "sagemaker",
        "gymnasium",
    ],
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
)
