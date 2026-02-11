import boto3
import json
import os
from botocore.exceptions import ClientError
from datetime import datetime
import io

class R2Storage:
    def __init__(self):
        self.account_id = os.environ.get("R2_ACCOUNT_ID")
        self.access_key = os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.environ.get("R2_BUCKET_NAME")

        if not all([self.account_id, self.access_key, self.secret_key, self.bucket_name]):
            print("Warning: R2 credentials not fully configured")
            self.client = None
        else:
            try:
                self.client = boto3.client(
                    's3',
                    endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
            except Exception as e:
                print(f"Failed to initialize R2 client: {e}")
                self.client = None

    def save_pdf(self, file_content: bytes, filename: str) -> str:
        if not self.client:
            return None

        try:
            key = f"pdfs/{filename}"
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_content,
                ContentType='application/pdf'
            )
            return key
        except ClientError as e:
            print(f"Error saving PDF to R2: {e}")
            return None

    def get_pdf(self, filename: str) -> bytes:
        if not self.client:
            return None

        try:
            key = f"pdfs/{filename}"
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            print(f"Error reading PDF from R2: {e}")
            return None

    def save_feedback(self, feedback_data: dict) -> bool:
        if not self.client:
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = feedback_data.get('filename', 'unknown').replace('.pdf', '')
            # Add random suffix to avoid collision
            import random
            suffix = random.randint(1000, 9999)
            key = f"feedback/{timestamp}_{safe_filename}_{suffix}.json"

            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(feedback_data),
                ContentType='application/json'
            )
            return True
        except ClientError as e:
            print(f"Error saving feedback to R2: {e}")
            return False
