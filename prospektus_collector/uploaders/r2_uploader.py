import time
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2Uploader:
    """Upload PDF ke Cloudflare R2 menggunakan boto3 (S3-compatible)."""

    def __init__(self, bucket: str, endpoint_url: str, access_key: str, secret_key: str):
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    @staticmethod
    def build_r2_key(source: str, emiten_code: str, filename: str) -> str:
        """Build R2 object key: prospektus/{source}/{emiten_code}/{filename}"""
        return f"prospektus/{source.lower()}/{emiten_code}/{filename}"

    def upload(self, pdf_bytes: bytes, r2_key: str, retries: int = 3) -> None:
        """
        Upload PDF bytes ke R2 dengan retry (exponential backoff).
        Raises RuntimeError setelah semua retry habis.
        Note: jika gagal, orchestrator menyimpan status 'failed' di tracker
        sehingga dapat di-retry saat run berikutnya.
        """
        for attempt in range(retries):
            try:
                self._client.put_object(
                    Bucket=self.bucket,
                    Key=r2_key,
                    Body=pdf_bytes,
                    ContentType="application/pdf",
                )
                logger.info(f"Uploaded {r2_key} ({len(pdf_bytes)} bytes)")
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"R2 upload attempt {attempt + 1}/{retries} failed: {e}. Retry in {wait}s")
                if attempt < retries - 1:
                    time.sleep(wait)
        raise RuntimeError(f"R2 upload failed after {retries} attempts for {r2_key}")

    def key_exists(self, r2_key: str) -> bool:
        """Cek apakah object sudah ada di R2."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=r2_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise
