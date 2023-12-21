import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def delete_all_objects(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    try:
        bucket.objects.delete()
        print(f"All objects in bucket '{bucket_name}' have been deleted.")
    except ClientError as e:
        print(f"An error occurred: {e}")


def upload_files_to_s3(bucket_name, directory):
    s3_client = boto3.client('s3')

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(subdir, file)
            try:
                s3_client.upload_file(full_path, bucket_name, full_path[len(directory):])
                print(f"File {full_path} uploaded successfully")
            except FileNotFoundError:
                print(f"The file {full_path} was not found")
            except NoCredentialsError:
                print("Credentials not available")


if __name__ == '__main__':
    bucket_name = 'wsp-kanawha-pilot-stac'
    directory = './stac/'
    delete_all_objects(bucket_name)
    upload_files_to_s3(bucket_name, directory)
