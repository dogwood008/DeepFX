
# coding: utf-8

# In[ ]:


import os
from google.cloud import storage


# In[ ]:


class GoogleCloudStorage:
    def __init__(self, bucket_name=os.environ['GOOGLE_CLOUD_STORAGE_BUCKET'],
                 service_account_json_path=os.environ['GOOGLE_SERVICE_ACCOUNT_JSON_PATH']):
        self.client = storage.Client.from_service_account_json(service_account_json_path)
        self.bucket = self.client.bucket(bucket_name)

    def models_path(self, path='models'):
        return path
    
    def upload(self, local_file_path, remote_dest_path):
        blob = self.bucket.blob(remote_dest_path)
        blob.upload_from_filename(filename=local_file_path)
    
    def upload_model(self, local_file_path):
        self.upload(local_file_path,
                      '%s/%s' % (self.models_path(), self._file_name(local_file_path)))

    def download(self, local_file_path, remote_src_path):
        blob = self.bucket.blob(remote_src_path)
        blob.download_to_filename(local_file_path)

    def download_model(self, local_file_path):
        self.download(local_file_path,
                      '%s/%s' % (self.models_path(), self._file_name(local_file_path)))

    def _file_name(self, file_path):
        return os.path.basename(file_path)


# In[ ]:


if __name__ == '__main__':
    gcs = GoogleCloudStorage()
    to_upload = True
    if to_upload:
        gcs.upload_model('README.md')
    else:
        gcs.download_model('README.md')

