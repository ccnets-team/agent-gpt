from sagemaker.estimator import Estimator
import re
from remote.one_click_params import SageMakerConfig, OneClickHyperparameters 

def training_validator(sage_config: SageMakerConfig, oneclick_params: OneClickHyperparameters):
    """Validate the SageMaker training job configuration."""
    if not sage_config.role_arn or not re.match(r"^arn:aws:iam::\d{12}:role/[\w+=,.@-]+", sage_config.role_arn):
        raise ValueError("Must provide a valid AWS IAM Role ARN.")
    if oneclick_params.env_id is None:
        raise ValueError("Must provide an environment ID.")
    if oneclick_params.env_url is None:
        raise ValueError("Must provide an environment URL.")

def one_click_sagemaker_training(sage_config: SageMakerConfig, oneclick_params: OneClickHyperparameters):
    """Launch a SageMaker training job for a one-click robotics environment."""
    training_validator(sage_config, oneclick_params)

    # Default values from SageMakerConfig + one-click hyperparameters
    hyperparams = {
        "env_id":      oneclick_params.env_id,
        "env_url":     oneclick_params.env_url,
        "output_path": sage_config.output_path,
    }
    estimator = Estimator(
        entry_point='train.py',
        role=sage_config.role_arn,
        instance_type=sage_config.instance_type,
        instance_count=sage_config.instance_count,
        output_path=sage_config.output_path,
        image_uri=sage_config.image_uri,
        max_run=sage_config.max_run,
        hyperparameters=hyperparams
    )

    print("[INFO] Final configuration:")
    for k, v in sage_config.items():
        print(f"  sage_config[{k}] = {v}")
    for k, v in hyperparams.items():
        print(f"  hyperparams[{k}] = {v}")

    # Start training
    estimator.fit()
    
def one_click_sagemaker_inference(sage_config: SageMakerConfig):

    # Possibly define hyperparams here, if needed
    hyperparams = {
        # inference-specific hyperparameters, if any
    }
    predictor  = Estimator(
        entry_point='infer.py',
        role=sage_config.role_arn,
        instance_type=sage_config.instance_type,
        instance_count=sage_config.instance_count,
        output_path=sage_config.output_path,
        image_uri=sage_config.image_uri,
        max_run=sage_config.max_run,
        hyperparameters=hyperparams
    )

    print("[INFO] Final configuration:")
    for k, v in sage_config.items():
        print(f"sage_config[{k}] = {v}")

    predictor.fit()