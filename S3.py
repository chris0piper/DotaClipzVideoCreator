import boto3
from botocore.exceptions import ClientError
import os

class S3:

    def __init__(self):
        self.s3_client = boto3.client('s3')


    def upload_file(self, file_name, bucket, object_name=None):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = os.path.basename(file_name)

        # Upload the file
        try:
            response = self.s3_client.upload_file(file_name, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def listFilesInBucket(self, bucketName):
        files = []
        for key in self.s3_client.list_objects(Bucket=bucketName)['Contents']:
            if '.' in key['Key']:
                files.append(key['Key'])
        return files
    
    def downloadFile(self, bucketName, filename, localFilename):

        self.s3_client.download_file(bucketName, filename, 'forYoutubeUpload/{}'.format(localFilename))

    def downloadNextAvailableFile(self, bucketname):
        # lists all objects in bucked
        filesToUpload = s3.listFilesInBucket(bucketname)

        # downloads the first of the list, remove the path
        outputFilename = ''
        if filesToUpload is not None and len(filesToUpload) > 0:
            fullPathToFile = filesToUpload[0]
            localFileame = fullPathToFile
            if '/' in fullPathToFile:
                localFileame = fullPathToFile.split('\\')[-1]
            s3.downloadFile(bucketname, fullPathToFile, localFileame)
            outputFilename = localFileame
        else:
            return None
        return outputFilename

# s3 = S3()

# print(s3.downloadNextAvailableFile('pending-youtube-upload'))
# exit()

# filesToUpload = s3.listFilesInBucket('pending-youtube-upload')
# if filesToUpload is not None and len(filesToUpload) > 0:
#     currentFileToUpload = filesToUpload[0]
#     s3.downloadFile('pending-youtube-upload', currentFileToUpload)
# print(currentFileToUpload)
# s3.upload_file('out.mp4', 'pending-youtube-upload')