import os
import csv
from pathlib import Path
import PyPDF2  # pip install pypdf2
import demo_util
from knowledge_storm.interface import log_execution_time
from demo_util import truncate_filename
from shutil import rmtree
from dotenv import dotenv_values
from minio.error import S3Error
from io import BytesIO, StringIO
from knowledge_storm.database import Topic, CSVDocument, PDFDocument
from sqlalchemy import desc

config = {**dotenv_values(".env"),
          **os.environ,}

minioHost = config["MINIO_HOST"]
minioUser = config["MINIO_ROOT_USER"]
minioPassword = config["MINIO_ROOT_PASSWORD"]
minioBucket = config["MINIO_BUCKET_NAME"]

def generate_url(file_path):
    base_url = "Storm_Interface/"
    return base_url + str(Path(file_path))

def clean_string(string):
    if string is None:
        return ''
    string = string.replace('"', '""')
    string = string.replace(';', ',')
    string = ' '.join(string.split())
    return string

def write_csv_file(data, csv_file_path, client, database, topic):
    print(f'Writing data to CSV file {csv_file_path}...')

    csv_buffer = StringIO()
    
    writer = csv.DictWriter(csv_buffer, fieldnames=['filename', 'url', 'title', 'author', 'description', 'content'], delimiter=';')
    writer.writeheader()
    for row in data:
        writer.writerow(row)

    csv_buffer.seek(0)
    
    byte_buffer = BytesIO(csv_buffer.getvalue().encode('utf-8'))

    try:
        client.put_object(
            minioBucket,
            csv_file_path,
            data = byte_buffer,
            length = byte_buffer.getbuffer().nbytes,
            content_type='text/csv'
        )
        print(f"Le fichier CSV '{csv_file_path}' a été généré avec succès.")
    except S3Error as err:
        print(f"Erreur lors de l'écriture du fichier csv dans MinIO : {err}")

    session = database.create_session()
    latest_topic = (
        session.query(Topic)
        .filter_by(topic=topic)
        .order_by(desc(Topic.start_date))
        .first()
    )

    json_conversation = CSVDocument(
        object_link = csv_file_path,
        topic_id = latest_topic.id
    )

    session.add(json_conversation)
    session.commit()
    session.refresh(json_conversation)
    session.close()

def extract_metadata_pdf(pdf_data):
    print("Extracting metadata...")
    try:
        reader = PyPDF2.PdfReader(pdf_data)
        metadata = reader.metadata
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}

def read_pdf(pdf_data):
    try:
        content = ""
        reader = PyPDF2.PdfReader(pdf_data)
        for page_num in range(min(10, len(reader.pages))):
            page = reader.pages[page_num]
            content += page.extract_text()
        return content
    except Exception as e:
        print(f"Error extracting content: {e}")
        return ""

def create_description(content, length=200):
    print("Creating description...")
    return content[:length].replace('\n', ' ') + "..." if len(content) > length else content.replace('\n', ' ')

def process_documents(directory, client):
    data = []
    print(f'Processing documents in directory {directory}...')
    try:
        objects = client.list_objects(minioBucket, prefix = directory, recursive = True)
        for obj in objects:
            if obj.object_name.endswith('.pdf'):

                response = client.get_object(minioBucket, obj.object_name)

                pdf_data = BytesIO(response.read())
                response.close()
                response.release_conn()

            
                title = obj.object_name.rsplit('/',1)[-1].replace('_', ' ')  # Title based on filename without extension
                metadata = extract_metadata_pdf(pdf_data)
                content = read_pdf(pdf_data)
                description = create_description(content)

                title = clean_string(title)
                content = clean_string(content)
                description = clean_string(description)

                content = str(content) if content else "Contenu non disponible"
                description = str(description) if description else "Description non disponible"

                data.append({
                    'filename': obj.object_name,
                    'url': obj.object_name,
                    'title': clean_string(metadata.get('title', 'Titre non disponible')),
                    'author': clean_string(metadata.get('author', 'Auteur non disponible')),
                    'description': clean_string(description),
                    'content': clean_string(content),
                })

        return data
    except S3Error as err:
        print(f"Erreur lors de l'accès au bicket {minioBucket} : {err}")

def create_new_article(topic, source_LLM, source_RM, csv_file_path, database, nbr_paragraphe, client, model = None):

    topic_name_cleaned = topic.replace(' ', '_').replace('/', '_')
    topic_name_truncated = truncate_filename(topic_name_cleaned)
    current_working_dir = topic_name_truncated
    
    log_execution_time(demo_util.set_storm_runner)
    if model : 
        runner = demo_util.set_storm_runner(source_LLM, source_RM, csv_file_path, current_working_dir, database, topic, client, model)
    else:
        runner = demo_util.set_storm_runner(source_LLM, source_RM, csv_file_path, current_working_dir, database, topic, client)

    
    database.update_status_topic(topic, "Lancement de STORM...")
    # STORM main gen outline
    runner.run(
        database=database,
        topic=topic,
        nbr_paragraphe=nbr_paragraphe,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True,
        client=client
    )
    
    # runner.run(topic=topic,
    #            do_research=False,
    #            do_generate_outline=False,
    #            do_generate_article=True,
    #            do_polish_article=True,
    #            remove_duplicate=False)

    runner.post_run(client = client, database = database)

async def create_csv_file(topic, files, client, database):
    
    print("Creating CSV file...")
    topic_name_cleaned = topic.replace(' ', '_').replace('/', '_')
    topic_name_truncated = truncate_filename(topic_name_cleaned)
    upload_directory = os.path.join(topic_name_truncated, "uploaded_folders")
    print(upload_directory)
    
    # Créer un répertoire pour sauvegarder les fichiers
    objects = client.list_objects(minioBucket, prefix=upload_directory, recursive=True)
    print(objects)
    for obj in objects:
        client.remove_object(minioBucket, obj.object_name)
    
    pdf_file_paths = []
    
    for file in files:
        filename_truncated = file.filename

        file_data = await file.read()
        file_stream = BytesIO(file_data)

        # Sauvegarder chaque fichier dans le répertoire
        file.filename = os.path.basename(filename_truncated)
        file_path = os.path.join(upload_directory, file.filename)
        client.put_object(minioBucket, file_path, data = file_stream, length = len(file_data))

        session = database.create_session()
        latest_topic = (
            session.query(Topic)
            .filter_by(topic=topic)
            .order_by(desc(Topic.start_date))
            .first()
        )

        json_conversation = PDFDocument(
            object_link = file_path,
            topic_id = latest_topic.id
        )

        session.add(json_conversation)
        session.commit()
        session.refresh(json_conversation)
        session.close()

        print(f"Saving file {file.filename} to {file_path}")
        #Mettre à jour bdd liée avec topic et mettre clé minio 
        pdf_file_paths.append(file_path)
        print(pdf_file_paths)

    # **Nouvelle section pour traiter les documents**
    data = process_documents(upload_directory, client)  # Assurez-vous d'appeler process_documents ici

    # Générer le nom du fichier CSV
    csv_file = f'generated_csv_file_{clean_string(topic)}.csv'
    print(f'CSV file {csv_file} will be created in {upload_directory}')
    csv_file_path = os.path.join(upload_directory, csv_file)

    # Écrire les données dans le fichier CSV
    write_csv_file(data, csv_file_path, client, database, topic)  # Passez le chemin complet du CSV

    return csv_file_path  # Retourne le chemin du fichier CSV