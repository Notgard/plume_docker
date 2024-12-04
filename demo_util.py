import base64
import datetime
import json
import os
import re
from dotenv import dotenv_values
import sys
from minio.error import S3Error

'''solution facile'''
# sys.path.append('./')

'''solution pérenne'''
# Obtenir le chemin du répertoire où se trouve le script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construire un chemin relatif par rapport au script
data_path = os.path.join(script_dir, '../../')
# Ajouter le chemin
sys.path.append(data_path)
###MODIF
config = {**dotenv_values(".env"), **dotenv_values(".env.docker"),
          **os.environ,}
print(config)
###
ollamaHost=config["OLLAMA_HOST"]
ollamaModel=config["OLLAMA_MODEL"]
aristoteAPIKey=config["ARISTOTE_API_KEY"]
aristoteURL=config["ARISTOTE_API_URL"]
ollamaPort=config["OLLAMA_PORT"]
qdrantStorage=config["QDRANT_STORAGE"]
minioBucket = config["MINIO_BUCKET_NAME"]


from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel, OllamaClient, VLLMClient
from knowledge_storm.rm import YouRM, VectorRM, BraveRM, ArxivRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from knowledge_storm.utils import QdrantVectorStoreManager, DemoTextProcessingHelper, truncate_filename, load_api_key
from knowledge_storm.database import DataBase

HF_HUB_OFFLINE = 1

class DemoFileIOHelper():
    @staticmethod
    def read_structure_to_dict(articles_root_path):
        """
        Reads the directory structure of articles stored in the given root path and
        returns a nested dictionary. The outer dictionary has article names as keys,
        and each value is another dictionary mapping file names to their absolute paths.

        Args:
            articles_root_path (str): The root directory path containing article subdirectories.

        Returns:
            dict: A dictionary where each key is an article name, and each value is a dictionary
                of file names and their absolute paths within that article's directory.
        """
        articles_dict = {}
        for topic_name in os.listdir(articles_root_path):
            topic_path = os.path.join(articles_root_path, topic_name)
            if os.path.isdir(topic_path):
                # Initialize or update the dictionary for the topic
                articles_dict[topic_name] = {}
                # Iterate over all files within a topic directory
                for file_name in os.listdir(topic_path):
                    file_path = os.path.join(topic_path, file_name)
                    articles_dict[topic_name][file_name] = os.path.abspath(file_path)
        return articles_dict

    @staticmethod
    def read_txt_file(file_path, client):
        """
        Reads the contents of a text file and returns it as a string.

        Args:
            file_path (str): The path to the text file to be read.

        Returns:
            str: The content of the file as a single string.
        """
        try:
            response = client.get_object(minioBucket, file_path)
            file_content = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            return file_content
        except S3Error as e:
            print(f"Error while reading the file: {e}")


    @staticmethod
    def read_json_file(file_path, client):
        """
        Reads a JSON file and returns its content as a Python dictionary or list,
        depending on the JSON structure.

        Args:
            file_path (str): The path to the JSON file to be read.

        Returns:
            dict or list: The content of the JSON file. The type depends on the
                        structure of the JSON file (object or array at the root).
        """
        try:
            response = client.get_object(minioBucket, file_path)
            file_content = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            return json.loads(file_content)
        except S3Error as e:
            print(f"Error while reading the file: {e}")

    @staticmethod
    def read_image_as_base64(image_path):
        """
        Reads an image file and returns its content encoded as a base64 string,
        suitable for embedding in HTML or transferring over networks where binary
        data cannot be easily sent.

        Args:
            image_path (str): The path to the image file to be encoded.

        Returns:
            str: The base64 encoded string of the image, prefixed with the necessary
                data URI scheme for images.
        """
        with open(image_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data)
        data = "data:image/png;base64," + encoded.decode("utf-8")
        return data

    @staticmethod
    def set_file_modification_time(file_path, modification_time_string):
        """
        Sets the modification time of a file based on a given time string in the California time zone.

        Args:
            file_path (str): The path to the file.
            modification_time_string (str): The desired modification time in 'YYYY-MM-DD HH:MM:SS' format.
        """
        california_tz = pytz.timezone('America/Los_Angeles')
        modification_time = datetime.datetime.strptime(modification_time_string, '%Y-%m-%d %H:%M:%S')
        modification_time = california_tz.localize(modification_time)
        modification_time_utc = modification_time.astimezone(datetime.timezone.utc)
        modification_timestamp = modification_time_utc.timestamp()
        os.utime(file_path, (modification_timestamp, modification_timestamp))

    @staticmethod
    def get_latest_modification_time(path):
        """
        Returns the latest modification time of all files in a directory in the California time zone as a string.

        Args:
            directory_path (str): The path to the directory.

        Returns:
            str: The latest file's modification time in 'YYYY-MM-DD HH:MM:SS' format.
        """
        california_tz = pytz.timezone('America/Los_Angeles')
        latest_mod_time = None

        file_paths = []
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        else:
            file_paths = [path]

        for file_path in file_paths:
            modification_timestamp = os.path.getmtime(file_path)
            modification_time_utc = datetime.datetime.utcfromtimestamp(modification_timestamp)
            modification_time_utc = modification_time_utc.replace(tzinfo=datetime.timezone.utc)
            modification_time_california = modification_time_utc.astimezone(california_tz)

            if latest_mod_time is None or modification_time_california > latest_mod_time:
                latest_mod_time = modification_time_california

        if latest_mod_time is not None:
            return latest_mod_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def assemble_article_data(article_file_path_dict):
        """
        Constructs a dictionary containing the content and metadata of an article
        based on the available files in the article's directory. This includes the
        main article text, citations from a JSON file, and a conversation log if
        available. The function prioritizes a polished version of the article if
        both a raw and polished version exist.

        Args:
            article_file_paths (dict): A dictionary where keys are file names relevant
                                    to the article (e.g., the article text, citations
                                    in JSON format, conversation logs) and values
                                    are their corresponding file paths.

        Returns:
            dict or None: A dictionary containing the parsed content of the article,
                        citations, and conversation log if available. Returns None
                        if neither the raw nor polished article text exists in the
                        provided file paths.
        """
        if "storm_gen_article.md" in article_file_path_dict or "storm_gen_article_polished.md" in article_file_path_dict:
            full_article_name = "storm_gen_article_polished.md" if "storm_gen_article_polished.md" in article_file_path_dict else "storm_gen_article.md"
            article_data = {"article": DemoTextProcessingHelper.parse(
                DemoFileIOHelper.read_txt_file(article_file_path_dict[full_article_name]))}
            if "url_to_info.json" in article_file_path_dict:
                article_data["citations"] = _construct_citation_dict_from_search_result(
                    DemoFileIOHelper.read_json_file(article_file_path_dict["url_to_info.json"]))
            if "conversation_log.json" in article_file_path_dict:
                article_data["conversation_log"] = DemoFileIOHelper.read_json_file(
                    article_file_path_dict["conversation_log.json"])
            return article_data
        return None

def _construct_citation_dict_from_search_result(search_results):
    if search_results is None:
        return None
    citation_dict = {}
    for url, index in search_results['url_to_unified_index'].items():
        citation_dict[index] = {'url': url,
                                'title': search_results['url_to_info'][url]['title'],
                                'snippets': search_results['url_to_info'][url]['snippets']}
    return citation_dict


def get_demo_dir():
    return os.path.dirname(os.path.abspath(__file__))

def set_storm_runner(
    source_LLM,
    source_RM,
    csv_file_path,
    current_working_dir,
    database,
    topic,
    client,
    model = ollamaModel
    ):
            
    # configure STORM runner
    llm_configs = STORMWikiLMConfigs()

    if source_LLM == "Aristote" :

        aristote_kwargs = {
            "api_key": aristoteAPIKey,
            "url": aristoteURL,
            "port": 443,
            "model_type": "text",
        }

        conv_simulator_lm = VLLMClient(model="casperhansen/llama-3-70b-instruct-awq",max_tokens=500, **aristote_kwargs)
        question_asker_lm = VLLMClient(model="casperhansen/llama-3-70b-instruct-awq",max_tokens=500, **aristote_kwargs)
        outline_gen_lm = VLLMClient(model="casperhansen/llama-3-70b-instruct-awq",max_tokens=400, **aristote_kwargs)
        article_gen_lm = VLLMClient(model="casperhansen/llama-3-70b-instruct-awq",max_tokens=700, **aristote_kwargs)
        article_polish_lm = VLLMClient(model="casperhansen/llama-3-70b-instruct-awq",max_tokens=4000, **aristote_kwargs)

    else :
        ###MODIF
        ollama_kwargs = {
            "model": model,
            "port": ollamaPort,
            "url": ollamaHost,
            "keep_alive": -1,
            "num_ctx" : 2048,
            "stop": ('\n\n---',)  # dspy uses "\n\n---" to separate examples. Open models sometimes generate this.
        }

        conv_simulator_lm = OllamaClient(max_tokens=500, **ollama_kwargs)
        question_asker_lm = OllamaClient(max_tokens=500, **ollama_kwargs)
        outline_gen_lm = OllamaClient(max_tokens=400, **ollama_kwargs)
        article_gen_lm = OllamaClient(max_tokens=700, **ollama_kwargs)
        article_polish_lm = OllamaClient(max_tokens=4000, **ollama_kwargs)

    llm_configs.set_conv_simulator_lm(conv_simulator_lm)
    llm_configs.set_question_asker_lm(question_asker_lm)
    llm_configs.set_outline_gen_lm(outline_gen_lm)
    llm_configs.set_article_gen_lm(article_gen_lm)
    llm_configs.set_article_polish_lm(article_polish_lm)
    
    engine_args = STORMWikiRunnerArguments(
        output_dir=current_working_dir,        
        max_conv_turn=1, #multiplie par autant le temps de run_conv
        max_perspective=3, #augmente légèrement le temps de run_conversation car parallélisé
        search_top_k=3, 
        retrieve_top_k=3,
        max_search_queries_per_turn=3,
        max_thread_num=16,
    )
    
    if source_RM == "local" :
    
        kwargs = {
            'file_path': csv_file_path,
            'content_column': 'content',
            'title_column': 'title',
            'url_column': 'url',
            'desc_column': 'description',
            'batch_size': 64,
            'vector_db_mode': qdrantStorage,
            'collection_name': 'my_documents',
            'embedding_model': "BAAI/bge-m3",
            'device': "cuda",
        }
        
        database.update_status_topic(topic, "Création de la base de donnée Qdrant...")
        QdrantVectorStoreManager.create_or_update_vector_store(
            client=client,
            vector_store_path='vector_store',
            **kwargs
            )
        rm = VectorRM(collection_name="my_documents", embedding_model="BAAI/bge-m3", device="cuda", k=engine_args.search_top_k)
        if qdrantStorage == "offline":
            rm.init_offline_vector_db(vector_store_path='./vector_store')
        else:
            rm.init_docker_vector_db()

    
    elif source_RM == "internet" :
        database.update_status_topic(topic, "Recherche sur Internet...")
        rm = BraveRM(brave_search_api_key=os.getenv('BRAVE_API_KEY'), k=engine_args.search_top_k)
        
    else :
        database.update_status_topic(topic, "Recherche sur Internet...")
        rm = ArxivRM(k=engine_args.search_top_k)
    
    
    runner = STORMWikiRunner(engine_args, llm_configs, rm)
    return runner