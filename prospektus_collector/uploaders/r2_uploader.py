"""Cloudflare R2 uploader for prospektus files."""
import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional, List

from ..config import (
    CLOUDFLARE_R2_BUCKET,
    CLOUDFLARE_R2_ENDPOINT,
    CLOUDFLARE_ACCESS_KEY_ID,
    CLOUDFLARE_SECRET_ACCESS_KEY
)


class R2Uploader:
    """Upload files to Cloudflare R2."""

    def __init__(self):
        self.bucket_name = CLOUDFLARE_R2_BUCKET
        self.endpoint_url = CLOUDFLARE_R2_ENDPOINT

        # Check if credentials are configured
        if not all([CLOUDFLARE_ACCESS_KEY_ID, CLOUDFLARE_SECRET_ACCESS_KEY, self.endpoint_url]):
            raise ValueError(
                "Cloudflare R2 credentials not configured. "
                "Set CLOUDFLARE_ACCESS_KEY_ID, CLOUDFLARE_SECRET_ACCESS_KEY, "
                "and CLOUDFLARE_R2_ENDPOINT environment variables."
            )

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=CLOUDFLARE_ACCESS_KEY_ID,
            aws_secret_access_key=CLOUDFLARE_SECRET_ACCESS_KEY,
            region_name='auto'
        )

    def upload_file(
        self,
        file_path: str,
        source: str,
        emiten_code: str,
        object_name: Optional[str] = None
    ) -> Optional[str]:
        """Upload a file to R2.

        Args:
            file_path: Local file path
            source: Source name (idx, ojk, ksei, emiten)
            emiten_code: Emiten stock code
            object_name: Custom R2 object name (optional)

        Returns:
            R2 key if successful, None otherwise
        """
        if object_name is None:
            filename = os.path.basename(file_path)
            object_name = f"prospektus/{source}/{emiten_code}/{filename}"

        try:
            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                object_name
            )
            print(f"✅ Uploaded to R2: {object_name}")
            return object_name
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            print(f"❌ Failed to upload to R2: {error_code} - {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to upload to R2: {e}")
            return None

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/pdf"
    ) -> Optional[str]:
        """Upload bytes directly to R2."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=data,
                ContentType=content_type
            )
            print(f"✅ Uploaded bytes to R2: {object_name}")
            return object_name
        except ClientError as e:
            print(f"❌ Failed to upload bytes to R2: {e}")
            return None

    def file_exists(self, source: str, emiten_code: str, filename: str) -> bool:
        """Check if a file already exists in R2."""
        object_name = f"prospektus/{source}/{emiten_code}/{filename}"

        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            return True
        except ClientError:
            return False

    def get_file_url(self, object_name: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned URL for a file."""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            print(f"Failed to generate URL: {e}")
            return None

    def list_files(
        self,
        source: Optional[str] = None,
        emiten_code: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[str]:
        """List files in R2 bucket."""
        prefix = "prospektus/"
        if source:
            prefix += f"{source}/"
        if emiten_code:
            prefix += f"{emiten_code}/"

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            print(f"Failed to list R2 files: {e}")
            return []

    def delete_file(self, object_name: str) -> bool:
        """Delete a file from R2."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            print(f"🗑️ Deleted from R2: {object_name}")
            return True
        except ClientError as e:
            print(f"Failed to delete from R2: {e}")
            return False

    def get_bucket_stats(self) -> dict:
        """Get bucket statistics."""
        try:
            # List all objects
            all_objects = []
            continuation_token = None

            while True:
                params = {'Bucket': self.bucket_name, 'MaxKeys': 1000}
                if continuation_token:
                    params['ContinuationToken'] = continuation_token

                response = self.s3_client.list_objects_v2(**params)
                all_objects.extend(response.get('Contents', []))

                if not response.get('IsTruncated'):
                    break
                continuation_token = response.get('NextContinuationToken')

            # Calculate stats
            total_size = sum(obj.get('Size', 0) for obj in all_objects)
            total_count = len(all_objects)

            # Group by prefix
            by_prefix = {}
            for obj in all_objects:
                key = obj['Key']
                parts = key.split('/')
                if len(parts) >= 2:
                    prefix = f"{parts[0]}/{parts[1]}"
                    by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

            return {
                "total_files": total_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "by_prefix": by_prefix
            }
        except ClientError as e:
            print(f"Failed to get bucket stats: {e}")
            return {}
