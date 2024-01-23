import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import argparse
import gzip


def upload_files_to_s3(bucket_name, directory: str, compress: bool = False):
    extra_args = {'ContentType': 'application/json'}
    if compress:
        extra_args.update({'ContentEncoding': 'gzip'})
    s3_client = boto3.client('s3')
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(subdir, file)
            try:
                if directory.endswith('/'):
                    directory_path_len = len(directory)
                else:
                    directory_path_len = len(directory) + 1    
                s3_path = full_path[directory_path_len:]
                with open(full_path, 'rb') as f:
                    if compress:
                        f = gzip.compress(f.read())
                    s3_client.put_object(Body=f, Bucket=bucket_name, Key=s3_path, **extra_args)
                print(f"File {full_path} uploaded successfully")
            except FileNotFoundError:
                print(f"The file {full_path} was not found")
            except NoCredentialsError:
                print("Credentials not available")


def main(directory: str, compress: bool = True):
    # copy directory recursively
    bucket_name = 'wsp-kanawha-pilot-stac'
    if compress:
        bucket_name = bucket_name + '-gzip'
    root_dir = os.listdir(directory)[0]
    upload_files_to_s3(bucket_name, directory, compress=compress)
    url = f"https://radiantearth.github.io/stac-browser/#/external/{bucket_name}.s3.amazonaws.com/{root_dir}/catalog.json"
    print(url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', help='directory to upload', default='./stac/')
    parser.add_argument('--gzip', help='gzip files', default=False, action='store_true')
    args = parser.parse_args()
    main(args.directory, args.gzip)
