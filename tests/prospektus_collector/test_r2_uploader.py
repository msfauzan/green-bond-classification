import pytest
from unittest.mock import MagicMock, patch
from prospektus_collector.uploaders.r2_uploader import R2Uploader


def test_r2uploader_builds_r2_key():
    uploader = R2Uploader.__new__(R2Uploader)
    key = uploader.build_r2_key(
        source="idx",
        emiten_code="ABCD",
        filename="20240101_ABCD_prospektus.pdf",
    )
    assert key == "prospektus/idx/ABCD/20240101_ABCD_prospektus.pdf"


def test_r2uploader_build_r2_key_lowercase_source():
    uploader = R2Uploader.__new__(R2Uploader)
    key = uploader.build_r2_key("IDX", "ABCD", "20240101_ABCD_file.pdf")
    assert key.startswith("prospektus/idx/")


@patch("prospektus_collector.uploaders.r2_uploader.boto3")
def test_r2uploader_upload_calls_put_object(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    uploader = R2Uploader(
        bucket="test-bucket",
        endpoint_url="https://test.r2.cloudflarestorage.com",
        access_key="key",
        secret_key="secret",
    )
    pdf_bytes = b"%PDF-1.4 test content"
    uploader.upload(pdf_bytes=pdf_bytes, r2_key="prospektus/idx/ABCD/file.pdf")
    mock_client.put_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="prospektus/idx/ABCD/file.pdf",
        Body=pdf_bytes,
        ContentType="application/pdf",
    )


@patch("prospektus_collector.uploaders.r2_uploader.boto3")
def test_r2uploader_upload_retries_on_failure(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.put_object.side_effect = [Exception("timeout"), Exception("timeout"), None]

    uploader = R2Uploader(
        bucket="test-bucket",
        endpoint_url="https://test.r2.cloudflarestorage.com",
        access_key="key",
        secret_key="secret",
    )
    uploader.upload(b"%PDF-1.4", r2_key="prospektus/idx/ABCD/file.pdf")
    assert mock_client.put_object.call_count == 3


@patch("prospektus_collector.uploaders.r2_uploader.boto3")
def test_r2uploader_key_exists_returns_true(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.head_object.return_value = {}

    uploader = R2Uploader(
        bucket="test-bucket",
        endpoint_url="https://test.r2.cloudflarestorage.com",
        access_key="key",
        secret_key="secret",
    )
    assert uploader.key_exists("prospektus/idx/ABCD/file.pdf") is True


@patch("prospektus_collector.uploaders.r2_uploader.boto3")
def test_r2uploader_key_exists_returns_false(mock_boto3):
    from botocore.exceptions import ClientError
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    uploader = R2Uploader(
        bucket="test-bucket",
        endpoint_url="https://test.r2.cloudflarestorage.com",
        access_key="key",
        secret_key="secret",
    )
    assert uploader.key_exists("prospektus/idx/ABCD/file.pdf") is False
