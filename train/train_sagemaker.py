# train_sagemaker.py (example file name)
import argparse
import re
from sagemaker.estimator import Estimator
from train.settings.sagemaker_config import SageMakerConfig

def train_sagemaker(sage_config: SageMakerConfig):
    """
    Launch a SageMaker training job for a one-click robotics environment.

    This function:
      1) Builds default config values from your provided env_id/env_url and SageMakerConfig.
      2) Sets up an ArgumentParser so CLI arguments can override those defaults.
      3) Validates the final configuration and logs it.
      4) Creates a SageMaker Estimator, passing leftover hyperparameters to train.py.
      5) Calls .fit() to start the training job on AWS.

    :param env_id:     Environment ID (e.g., 'Humanoid-v4').
    :param env_url:    Environment URL/endpoint (e.g., 'http://127.0.0.1:5000').
    :param sage_config: SageMakerConfig instance (role ARN, instance_type, etc.).
    """

    # 1. Build a base config from function arguments + SageMakerConfig
    default_config = {
        "env_id": sage_config.env_id,
        "env_url": sage_config.env_url,
        "role_arn": sage_config.role_arn,
        "output_path": sage_config.output_path,
        "instance_type": sage_config.instance_type,
        "instance_count": sage_config.instance_count,
        "max_run": sage_config.max_run,
        "image_uri": sage_config.image_uri,
    }

    # 2. Parse CLI arguments to allow overrides
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker training job for one-click robotics."
    )
    for key, value in default_config.items():
        # If value is None, default to str for possible CLI overrides
        arg_type = type(value) if value is not None else str
        parser.add_argument(
            f"--{key}",
            type=arg_type,
            default=value,
            help=f"Set {key} (default: {value})"
        )

    args = parser.parse_args()
    hyperparams = vars(args)

    # 3. Basic validation: Must have a valid role_arn
    if not hyperparams.get("role_arn"):
        raise ValueError("A valid AWS IAM Role ARN is required (use '--role_arn' or set in SageMakerConfig).")

    # (Optional) Validate S3 output path format
    # s3_pattern = r"^s3://[a-zA-Z0-9.\-_]+(/[a-zA-Z0-9.\-_]+)*$"
    # if not re.match(s3_pattern, hyperparams['output_path']):
    #     raise ValueError("Invalid S3 output path format. Check the 'output_path' value.")

    # 4. Extract SageMaker parameters; pass everything else as hyperparameters
    final_role_arn = hyperparams["role_arn"]
    final_output_path = hyperparams["output_path"]
    final_instance_type = hyperparams["instance_type"]
    final_instance_count = int(hyperparams["instance_count"])
    final_max_run = int(hyperparams["max_run"])
    final_image_uri = hyperparams["image_uri"]

    # Remove these from hyperparams so the rest become custom hyperparameters
    for key_to_remove in ["role_arn", "output_path", "instance_type",
                          "instance_count", "max_run", "image_uri"]:
        del hyperparams[key_to_remove]

    # 5. Create the Estimator
    estimator = Estimator(
        entry_point='train.py',
        role=final_role_arn,
        instance_type=final_instance_type,
        instance_count=final_instance_count,
        output_path=final_output_path,
        image_uri=final_image_uri,
        hyperparameters=hyperparams,  # everything left is passed to train.py
        max_run=final_max_run
    )

    # 6. Log final config for clarity
    print("[INFO] Launching SageMaker training with the following configuration:")
    print(f"  role_arn       : {final_role_arn}")
    print(f"  output_path    : {final_output_path}")
    print(f"  instance_type  : {final_instance_type}")
    print(f"  instance_count : {final_instance_count}")
    print(f"  max_run        : {final_max_run} sec")
    print(f"  image_uri      : {final_image_uri}")

    # Show leftover hyperparameters (env_id, env_url, or anything else)
    print("[INFO] Additional Hyperparameters (passed to train.py):")
    for k, v in hyperparams.items():
        print(f"  {k} = {v}")

    # 7. Launch the training job on SageMaker
    estimator.fit()
