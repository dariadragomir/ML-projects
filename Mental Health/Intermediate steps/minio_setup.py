from minio import Minio
from minio.error import S3Error

from minio import Minio
from minio.error import S3Error

minio_client = Minio(
    "localhost:9000", 
    access_key="minioadmin", 
    secret_key="minioadmin", 
    secure=False
)

# create bucket
bucket_name = "datasets"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created.")
else:
    print(f"Bucket '{bucket_name}' already exists.")

# post request
file_path = "/Users/dariadragomir/AI_siemens/mental health/merged_file.csv" 
object_name = "merged_file.csv" 
minio_client.fput_object(bucket_name, object_name, file_path)
print(f"File '{object_name}' uploaded to bucket '{bucket_name}'.")

# get request
download_path = "/Users/dariadragomir/AI_siemens/mental health/downloaded_merged_file.csv" 
minio_client.fget_object(bucket_name, object_name, download_path)
print(f"File '{object_name}' downloaded to '{download_path}'.")


objects = minio_client.list_objects(bucket_name)
for obj in objects:
    print(f"Object: {obj.object_name}")
