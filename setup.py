from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    
setup(
    name="agent-gpt-aws",
    version="0.3.3",
    packages=find_packages(), 
    entry_points={
        "console_scripts": [
            "agent-gpt=agent_gpt.cli:app",
        ],
    },
    install_requires=requirements,    
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
