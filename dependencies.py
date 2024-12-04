from fastapi.templating import Jinja2Templates
from dotenv import dotenv_values
import os
import mariadb # type: ignore
import sys
import requests # type: ignore
import markdown # type: ignore
from minio.error import S3Error
from minio import Minio

from knowledge_storm.database import DataBase

config = {**dotenv_values(".env"), **dotenv_values(".env.docker"),
          **os.environ,}
ollamaHost=config["OLLAMA_HOST"]
ollamaPort=config["OLLAMA_PORT"]
mariadbHost = config["MARIADB_HOST"]
mariadbPassword = config["MYSQL_ROOT_PASSWORD"]
mariadbDb = config["MYSQL_DATABASE"]
minioBucket = config["MINIO_BUCKET_NAME"]
minioUser = config["MINIO_ROOT_USER"]
minioHost = config["MINIO_HOST"]
minioPassword = config["MINIO_ROOT_PASSWORD"]


templates = Jinja2Templates(directory="templates")

try:
    database = DataBase(mariadbHost, mariadbPassword, mariadbDb, port=3306)
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

try:
    client = Minio(minioHost, access_key=minioUser, secret_key=minioPassword, secure = False)
    print("Connection réussie")
except S3Error as exc:
    print(f"Error connecting to MinIO: {exc}")
    sys.exit(1)

try:
    found = client.bucket_exists(minioBucket)
except S3Error as e:
    print(f"Error searching for bucket : {e}")
    sys.exit(1)
print(found)

if not found:
    client.make_bucket(minioBucket)
    print("Created Bucket", minioBucket)
else:
    print("Bucker", minioBucket, "already exists")

templates = Jinja2Templates(directory="templates")
 

def get_models():
    url = ollamaHost + ":" + ollamaPort +"/v1/models"
    response = requests.get(url)
    if response.status_code ==200:
        li = []
        for i in response.json()['data']:
            model = i['id']
            j=0
            model_without_version = ""
            while model[j] != ":":
                model_without_version+=model[j]
                j+=1
            li.append(model_without_version)
        return(li)
    else:
        print(f"Erreur  : {response.status_code}")



def get_file (filename, client):
     
    file_content = ""
    if filename:
        file_path = os.path.join(filename.replace(" ", "_"), 'productions', 'storm_gen_article_polished.md')
        try:
            response = client.get_object(minioBucket, file_path)
            file_content = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            print(file_content)
        except S3Error as e:
            print(f"Error while reading the file: {e}")

        file_content = markdown.markdown(file_content)
        file_content = file_content.replace('<h4>', '<h6>').replace('</h4>', '</h6>')
        file_content = file_content.replace('<h3>', '<h5>').replace('</h3>', '</h5>')
        file_content = file_content.replace('<h2>', '<h4>').replace('</h2>', '</h4>')
        file_content = file_content.replace('<h1>', '<h3>').replace('</h1>', '</h3>')
        file_content = file_content.replace('<a', '<a target="_blank"')
        print(file_content)
        return file_content