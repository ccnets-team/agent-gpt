from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-gpt",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["cli"],
    install_requires=[
        "typer",
        "pyyaml",
        "boto3",
        "uvicorn",
        "fastapi",
        "sagemaker",
        "gymnasium",
        # Add other dependencies as needed...
    ],
    entry_points={
        "console_scripts": [
            "agent-gpt=cli:app",
        ],
    },
    author="Your Name",
    description="AgentGPT CLI for training and inference on AWS SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
