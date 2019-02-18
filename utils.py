from google.cloud import storage
from urllib.parse import urlparse

def gcs_upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def gcs_bucket_prefix(full_path):
    parsed_url = urlparse(full_path)
    bucket     = parsed_url.netloc
    prefix     = parsed_url.path[1:]
    return bucket, prefix