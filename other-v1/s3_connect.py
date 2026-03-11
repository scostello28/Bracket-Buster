import pandas as pd
import boto3


s3 = boto3.client("s3")

bucket_name = "bracket-buster"

'''Create Bucket'''
# try:
#     s3.create_bucket(
#     Bucket=bucket_name,
#     CreateBucketConfiguration={'LocationConstraint': 'us-west-2'},
# )
# except Exception as e:
#     print(e)  # Note: BucketAlreadyOwnedByYou means you already created the bucket.

'''list my buckets'''
response = s3.list_buckets()
# print(response)
buckets = response['Buckets']
# print(buckets)
# print([bucket['Name'] for bucket in buckets])

'''Upload data files'''
file_name = 'final_model_data/gamelog_exp_clust.pkl'
remote_pathname = 'gamelog_exp_clust.pkl'

s3.upload_file(
    Bucket=bucket_name,
    Filename=file_name,
    Key=remote_pathname
    )

'''Show items in bucket'''
# response = s3.list_objects(
#     Bucket=bucket_name
# )
#
# print(response['Contents'])
# print([item['Key'] for item in response['Contents']])
