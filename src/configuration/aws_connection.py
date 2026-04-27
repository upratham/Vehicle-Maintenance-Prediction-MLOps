import boto3

from src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME


class S3Client:
    """Process-cached boto3 S3 resource and client."""

    s3_client = None
    s3_resource = None

    def __init__(self, region_name: str = REGION_NAME) -> None:
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            if not AWS_ACCESS_KEY_ID_ENV_KEY:
                raise EnvironmentError("Environment variable AWS_ACCESS_KEY_ID is not set.")
            if not AWS_SECRET_ACCESS_KEY_ENV_KEY:
                raise EnvironmentError("Environment variable AWS_SECRET_ACCESS_KEY is not set.")
            creds = dict(
                aws_access_key_id=AWS_ACCESS_KEY_ID_ENV_KEY,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY_ENV_KEY,
                region_name=region_name,
            )
            S3Client.s3_resource = boto3.resource("s3", **creds)
            S3Client.s3_client = boto3.client("s3", **creds)
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
