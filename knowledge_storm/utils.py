import concurrent.futures
import json
import logging
import os
import pickle
import re
import regex
import sys
import time
from typing import List, Dict
# import mariadb

import httpx
import pandas as pd
import toml
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from dotenv import dotenv_values
from trafilatura import extract
from minio.error import S3Error
from io import BytesIO

from .lm import OpenAIModel

logging.getLogger("httpx").setLevel(logging.WARNING)  # Disable INFO logging for httpx.

import requests
import fitz
import threading

config = {**dotenv_values(".env"),
          **os.environ,}
qdrantPort = config["QDRANT_PORT"]
qdrantHost = config["QDRANT_HOST"]
minioBucket = config["MINIO_BUCKET_NAME"]
minioHost = config["MINIO_HOST"]


def truncate_filename(filename, max_length=125):
    """Truncate filename to max_length to ensure the filename won't exceed the file system limit.

    Args:
        filename: str
        max_length: int, default to 125 (usual path length limit is 255 chars)
    """

    if len(filename) > max_length:
        truncated_filename = filename[:max_length]
        logging.warning(
            f"Filename is too long. Filename is truncated to {truncated_filename}."
        )
        return truncated_filename

    return filename


def load_api_key(toml_file_path):
    try:
        with open(toml_file_path, "r") as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)


def makeStringRed(message):
    return f"\033[91m {message}\033[00m"


class QdrantVectorStoreManager:
    """
    Helper class for managing the Qdrant vector store, can be used with `VectorRM` in rm.py.

    Before you initialize `VectorRM`, call `create_or_update_vector_store` to create or update the vector store.
    Once you have the vector store, you can initialize `VectorRM` with the vector store path or the Qdrant server URL.
    """

    @staticmethod
    def _check_create_collection(
        client: QdrantClient, collection_name: str, model: HuggingFaceEmbeddings
    ):
        """Check if the Qdrant collection exists and create it if it does not."""
        if client is None:
            raise ValueError("Qdrant client is not initialized.")
        if client.collection_exists(collection_name=f"{collection_name}"):
            client.delete_collection(collection_name=f"{collection_name}")
        print(
            f"Collection {collection_name} does not exist. Creating the collection..."
        )
        # create the collection
        client.create_collection(
            collection_name=f"{collection_name}",
            vectors_config=models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
        )
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=model,
        )

    @staticmethod
    def _init_online_vector_db(
        url: str, api_key: str, collection_name: str, model: HuggingFaceEmbeddings
    ):
        """Initialize the Qdrant client that is connected to an online vector store with the given URL and API key.

        Args:
            url (str): URL of the Qdrant server.
            api_key (str): API key for the Qdrant server.
        """
        if api_key is None:
            if not os.getenv("QDRANT_API_KEY"):
                raise ValueError("Please provide an api key.")
            api_key = os.getenv("QDRANT_API_KEY")
        if url is None:
            raise ValueError("Please provide a url for the Qdrant server.")

        try:
            client = QdrantClient(url=url, api_key=api_key)
            return QdrantVectorStoreManager._check_create_collection(
                client=client, collection_name=collection_name, model=model
            )
        except Exception as e:
            raise ValueError(f"Error occurs when connecting to the server: {e}")

    @staticmethod
    def _init_offline_vector_db(
        vector_store_path: str, collection_name: str, model: HuggingFaceEmbeddings
    ):
        """Initialize the Qdrant client that is connected to an offline vector store with the given vector store folder path.

        Args:
            vector_store_path (str): Path to the vector store.
        """
        if vector_store_path is None:
            raise ValueError("Please provide a folder path.")

        try:
            client = QdrantClient(path=vector_store_path)
            return QdrantVectorStoreManager._check_create_collection(
                client=client, collection_name=collection_name, model=model
            )
        except Exception as e:
            raise ValueError(f"Error occurs when loading the vector store: {e}")

    @staticmethod 
    def _init_docker_vector_db(host : str, port : int, collection_name : str, model : HuggingFaceEmbeddings):
        """
        Initialize the Qdrant client that is connected to an online vector store with the given URL and API key.

        Args:
            url (str): URL of the Qdrant server.
            api_key (str): API key for the Qdrant server.
        """
        try:
            client = QdrantClient(host = host, port = port)
            return QdrantVectorStoreManager._check_create_collection(
                client=client, collection_name=collection_name, model=model
            )
        except Exception as e:
            raise ValueError(f"Error occurs when loading the vector store: {e}")


    @staticmethod
    def create_or_update_vector_store(
        collection_name: str,
        vector_db_mode: str,
        file_path: str,
        content_column: str,
        client,
        title_column: str = "title",
        url_column: str = "url",
        desc_column: str = "description",
        batch_size: int = 64,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        vector_store_path: str = None,
        url: str = None,
        qdrant_api_key: str = None,
        embedding_model: str = "BAAI/bge-m3",
        device: str = "cpu",
        host = qdrantHost,
        port = qdrantPort
    ):
        """
        Takes a CSV file and adds each row in the CSV file to the Qdrant collection.

        This function expects each row of the CSV file as a document.
        The CSV file should have columns for "content", "title", "URL", and "description".

        Args:
            collection_name: Name of the Qdrant collection.
            vector_store_path (str): Path to the directory where the vector store is stored or will be stored.
            vector_db_mode (str): Mode of the Qdrant vector store (offline or online).
            file_path (str): Path to the CSV file.
            content_column (str): Name of the column containing the content.
            title_column (str): Name of the column containing the title. Default is "title".
            url_column (str): Name of the column containing the URL. Default is "url".
            desc_column (str): Name of the column containing the description. Default is "description".
            batch_size (int): Batch size for adding documents to the collection.
            chunk_size: Size of each chunk if you need to build the vector store from documents.
            chunk_overlap: Overlap between chunks if you need to build the vector store from documents.
            embedding_model: Name of the Hugging Face embedding model.
            device: Device to run the embeddings model on, can be "mps", "cuda", "cpu".
            qdrant_api_key: API key for the Qdrant server (Only required if the Qdrant server is online).
        """
        # check if the collection name is provided
        if collection_name is None:
            raise ValueError("Please provide a collection name.")

        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        if file_path is None:
            raise ValueError("Please provide a file path.")
        # check if the file is a csv file
        if not file_path.endswith(".csv"):
            raise ValueError(f"Not valid file format. Please provide a csv file.")
        if content_column is None:
            raise ValueError("Please provide the name of the content column.")
        if url_column is None:
            raise ValueError("Please provide the name of the url column.")

        # try to initialize the Qdrant client
        qdrant = None
        if vector_db_mode == "online":
            qdrant = QdrantVectorStoreManager._init_online_vector_db(
                url=url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                model=model,
            )
        elif vector_db_mode == "offline":
            qdrant = QdrantVectorStoreManager._init_offline_vector_db(
                vector_store_path=vector_store_path,
                collection_name=collection_name,
                model=model,
            )
        elif vector_db_mode == "docker":
            qdrant = QdrantVectorStoreManager._init_docker_vector_db(
                host = host,
                port = port,
                collection_name=collection_name,
                model = model
            )
        else:
            raise ValueError(
                "Invalid vector_db_mode. Please provide either 'online' or 'offline'."
            )
        if qdrant is None:
            raise ValueError("Qdrant client is not initialized.")
        
        try:
            response = client.get_object(minioBucket, file_path)
        except S3Error as e:
            print(f"Impossible de récupérer le csv {e}")

        
        # read the csv file
        df = pd.read_csv(BytesIO(response.data), sep = ';')
        response.close()
        response.release_conn()
        
        # check that content column exists and url column exists
        if content_column not in df.columns:
            raise ValueError(
                f"Content column {content_column} not found in the csv file."
            )
        if url_column not in df.columns:
            raise ValueError(f"URL column {url_column} not found in the csv file.")

        documents = [
            Document(
                page_content=row[content_column],
                metadata={
                    "title": row.get(title_column, ""),
                    "url": row[url_column],
                    "description": row.get(desc_column, ""),
                },
            )
            for row in df.to_dict(orient="records")
        ]

        # split the documents
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )
        split_documents = text_splitter.split_documents(documents)

        # update and save the vector store
        num_batches = (len(split_documents) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(split_documents))
            qdrant.add_documents(
                documents=split_documents[start_idx:end_idx],
                batch_size=batch_size,
            )

        # close the qdrant client
        qdrant.client.close()


class ArticleTextProcessing:
    @staticmethod
    def limit_word_count_preserve_newline(input_string, max_word_count):
        """
        Limit the word count of an input string to a specified maximum, while preserving the integrity of complete lines.

        The function truncates the input string at the nearest word that does not exceed the maximum word count,
        ensuring that no partial lines are included in the output. Words are defined as text separated by spaces,
        and lines are defined as text separated by newline characters.

        Args:
            input_string (str): The string to be truncated. This string may contain multiple lines.
            max_word_count (int): The maximum number of words allowed in the truncated string.

        Returns:
            str: The truncated string with word count limited to `max_word_count`, preserving complete lines.
        """

        word_count = 0
        limited_string = ""

        for word in input_string.split("\n"):
            line_words = word.split()
            for lw in line_words:
                if word_count < max_word_count:
                    limited_string += lw + " "
                    word_count += 1
                else:
                    break
            if word_count >= max_word_count:
                break
            limited_string = limited_string.strip() + "\n"

        return limited_string.strip()

    @staticmethod
    def remove_citations(s):
        """
        Removes all citations from a given string. Citations are assumed to be in the format
        of numbers enclosed in square brackets, such as [1], [2], or [1, 2], etc. This function searches
        for all occurrences of such patterns and removes them, returning the cleaned string.

        Args:
            s (str): The string from which citations are to be removed.

        Returns:
            str: The string with all citation patterns removed.
        """

        return re.sub(r"\[\d+(?:,\s*\d+)*\]", "", s)

    @staticmethod
    def parse_citation_indices(s):
        """
        Extracts citation indexes from the provided content string and returns them as a list of integers.

        Args:
            content (str): The content string containing citations in the format [number].

        Returns:
            List[int]: A list of unique citation indexes extracted from the content, in the order they appear.
        """
        matches = re.findall(r"\[\d+\]", s)
        return [int(index[1:-1]) for index in matches]

    @staticmethod
    def remove_uncompleted_sentences_with_citations(text):
        """
        Removes uncompleted sentences and standalone citations from the input text. Sentences are identified
        by their ending punctuation (.!?), optionally followed by a citation in square brackets (e.g., "[1]").
        Grouped citations (e.g., "[1, 2]") are split into individual ones (e.g., "[1] [2]"). Only text up to
        and including the last complete sentence and its citation is retained.

        Args:
            text (str): The input text from which uncompleted sentences and their citations are to be removed.

        Returns:
            str: The processed string with uncompleted sentences and standalone citations removed, leaving only
            complete sentences and their associated citations if present.
        """

        # Convert citations like [1, 2, 3] to [1][2][3].
        def replace_with_individual_brackets(match):
            numbers = match.group(1).split(", ")
            return " ".join(f"[{n}]" for n in numbers)

        # Deduplicate and sort individual groups of citations.
        def deduplicate_group(match):
            citations = match.group(0)
            unique_citations = list(set(re.findall(r"\[\d+\]", citations)))
            sorted_citations = sorted(
                unique_citations, key=lambda x: int(x.strip("[]"))
            )
            # Return the sorted unique citations as a string
            return "".join(sorted_citations)

        text = re.sub(r"\[([0-9, ]+)\]", replace_with_individual_brackets, text)
        text = re.sub(r"(\[\d+\])+", deduplicate_group, text)

        # Deprecated: Remove sentence without proper ending punctuation and citations.
        # Split the text into sentences (including citations).
        # sentences_with_trailing = re.findall(r'([^.!?]*[.!?].*?)(?=[^.!?]*[.!?]|$)', text)

        # Filter sentences to ensure they end with a punctuation mark and properly formatted citations
        # complete_sentences = []
        # for sentence in sentences_with_trailing:
        #     # Check if the sentence ends with properly formatted citations
        #     if re.search(r'[.!?]( \[\d+\])*$|^[^.!?]*[.!?]$', sentence.strip()):
        #         complete_sentences.append(sentence.strip())

        # combined_sentences = ' '.join(complete_sentences)

        # Check for and append any complete citations that follow the last sentence
        # trailing_citations = re.findall(r'(\[\d+\]) ', text[text.rfind(combined_sentences) + len(combined_sentences):])
        # if trailing_citations:
        #     combined_sentences += ' '.join(trailing_citations)

        # Regex pattern to match sentence endings, including optional citation markers.
        eos_pattern = r"([.!?])\s*(\[\d+\])?\s*"
        matches = list(re.finditer(eos_pattern, text))
        if matches:
            last_match = matches[-1]
            text = text[: last_match.end()].strip()

        return text

    @staticmethod
    def clean_up_citation(conv):
        for turn in conv.dlg_history:
            if "References:" in turn.agent_utterance:
                turn.agent_utterance = turn.agent_utterance[
                    : turn.agent_utterance.find("References:")
                ]
            if "Sources:" in turn.agent_utterance:
                turn.agent_utterance = turn.agent_utterance[
                    : turn.agent_utterance.find("Sources:")
                ]
            turn.agent_utterance = turn.agent_utterance.replace("Answer:", "").strip()
            try:
                max_ref_num = max(
                    [int(x) for x in re.findall(r"\[(\d+)\]", turn.agent_utterance)]
                )
            except Exception as e:
                max_ref_num = 0
            if max_ref_num > len(turn.search_results):
                for i in range(len(turn.search_results), max_ref_num + 1):
                    turn.agent_utterance = turn.agent_utterance.replace(f"[{i}]", "")
            turn.agent_utterance = (
                ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                    turn.agent_utterance
                )
            )

        return conv

    @staticmethod
    def clean_up_outline(outline, topic=""):
        
        print("1", outline)
        
        outline = re.sub(r'\b\d+\.\d+\.\d+\.\d', '####', outline)
        outline = re.sub(r'\b\d+\.\d+\.\d', '###', outline)
        outline = re.sub(r'\b\d+\.\d', '##', outline)
        outline = re.sub(r'\b\d+\.', '#', outline)
        outline = outline.replace('#### ####', '####')
        outline = outline.replace('### ###', '###')
        outline = outline.replace('## ##', '##')
        outline = outline.replace('# #', '#')
        
        output_lines = []
        current_level = 0  # To track the current section level

        for line in outline.split("\n"):
            stripped_line = line.strip()

            if topic != "" and f"# {topic.lower()}" in stripped_line.lower():
                output_lines = []

            # Check if the line is a section header
            if stripped_line.startswith("#"):
                current_level = stripped_line.count("#")
                output_lines.append(stripped_line)
            # Check if the line is a bullet point
            elif stripped_line.startswith("-"):
                subsection_header = (
                    "#" * (current_level + 1) + " " + stripped_line[1:].strip()
                )
                output_lines.append(subsection_header)

        outline = "\n".join(output_lines)
        
        print("2", outline)
        
        outline = re.sub(r"#[#]? --.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Number of sections.*?(?=##|$)", "", outline, flags=re.DOTALL)

        # Remove references.
        outline = re.sub(r"#[#]? See also.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? See Also.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Notes.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? References.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(
            r"#[#]? External links.*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(
            r"#[#]? External Links.*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(r"#[#]? Bibliography.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(
            r"#[#]? Further reading*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(
            r"#[#]? Further Reading*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(r"#[#]? Summary.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Appendices.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Appendix.*?(?=##|$)", "", outline, flags=re.DOTALL)
        
        # Supprimer les références.
        outline = re.sub(r"#[#]? Voir aussi.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Voir Aussi.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Notes.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Références.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(
            r"#[#]? Liens externes.*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(
            r"#[#]? Liens Externes.*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(r"#[#]? Bibliographie.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(
            r"#[#]? Lectures complémentaires.*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(
            r"#[#]? Lectures Complémentaires.*?(?=##|$)", "", outline, flags=re.DOTALL
        )
        outline = re.sub(r"#[#]? Sommaire.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Résumé.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Annexes.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Annexe.*?(?=##|$)", "", outline, flags=re.DOTALL)
        
        # clean up citation in outline
        outline = re.sub(r"\[.*?\]", "", outline)
        
        print("3", outline)
                
        # Cherche toutes les occurrences de "# " qui ne sont pas précédées d'un "#"
        sections = re.findall(r"(?<!#)^# .+", outline, flags=re.MULTILINE)

        # Affiche le nombre de sections trouvées
        print("Nombre de sections de niveau 1 trouvées :", len(sections))

        # Vérifie s'il n'y a qu'une seule occurrence de section de niveau 1
        if len(sections) == 1:
            # Supprime la section unique de niveau 1
            outline = re.sub(r"(?<!#)^# .+\n(.*\n)*?(?=^#|$)", "", outline, flags=re.MULTILINE)
        
        # print("4", outline)
        return outline

    @staticmethod
    def clean_up_section(text):
        """Clean up a section:
        1. Remove uncompleted sentences (usually due to output token limitation).
        2. Deduplicate individual groups of citations.
        3. Remove unnecessary summary."""
        
        # post treatment in case of the use of Ollama could be useful

        paragraphs = text.split("\n")
        output_paragraphs = []
        summary_sec_flag = False
        for p in paragraphs:
            p = p.strip()
            if len(p) == 0:
                continue
            if not p.startswith("#"):
                p = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(p)
            if summary_sec_flag:
                if p.startswith("#"):
                    summary_sec_flag = False
                else:
                    continue
            if (
                p.startswith("Overall")
                or p.startswith("In summary")
                or p.startswith("In conclusion")
            ):
                continue
            if "# Summary" in p or "# Conclusion" in p:
                summary_sec_flag = True
                continue
            output_paragraphs.append(p)

        # Join with '\n\n' for markdown format.
        return "\n\n".join(output_paragraphs)

    @staticmethod
    def update_citation_index(s, citation_map):
        """Update citation index in the string based on the citation map."""
        for original_citation in citation_map:
            s = s.replace(
                f"[{original_citation}]", f"__PLACEHOLDER_{original_citation}__"
            )
        for original_citation, unify_citation in citation_map.items():
            s = s.replace(f"__PLACEHOLDER_{original_citation}__", f"[{unify_citation}]")

        return s

    @staticmethod
    def parse_article_into_dict(input_string):
        """
        Parses a structured text into a nested dictionary. The structure of the text
        is defined by markdown-like headers (using '#' symbols) to denote sections
        and subsections. Each section can contain content and further nested subsections.

        The resulting dictionary captures the hierarchical structure of sections, where
        each section is represented as a key (the section's title) mapping to a value
        that is another dictionary. This dictionary contains two keys:
        - 'content': content of the section
        - 'subsections': a list of dictionaries, each representing a nested subsection
        following the same structure.

        Args:
            input_string (str): A string containing the structured text to parse.

        Returns:
            A dictionary representing contains the section title as the key, and another dictionary
        as the value, which includes the 'content' and 'subsections' keys as described above.
        """
        lines = input_string.split("\n")
        lines = [line for line in lines if line.strip()]
        root = {"content": "", "subsections": {}}
        current_path = [(root, -1)]  # (current_dict, level)

        for line in lines:
            if line.startswith("#"):
                level = line.count("#")
                title = line.strip("# ").strip()
                new_section = {"content": "", "subsections": {}}

                # Pop from stack until find the parent level
                while current_path and current_path[-1][1] >= level:
                    current_path.pop()

                # Append new section to the nearest upper level's subsections
                current_path[-1][0]["subsections"][title] = new_section
                current_path.append((new_section, level))
            else:
                current_path[-1][0]["content"] += line + "\n"

        return root["subsections"]


class FileIOHelper:
    @staticmethod
    def dump_json(obj, file_name, encoding="utf-8"):
        with open(file_name, "w", encoding=encoding) as fw:
            json.dump(obj, fw, default=FileIOHelper.handle_non_serializable)

    @staticmethod
    def handle_non_serializable(obj):
        return "non-serializable contents"  # mark the non-serializable part

    @staticmethod
    def load_json(file_name, encoding="utf-8"):
        with open(file_name, "r", encoding=encoding) as fr:
            return json.load(fr)

    @staticmethod
    def write_str(s, path):
        with open(path, "w") as f:
            f.write(s)

    @staticmethod
    def load_str(path):
        with open(path, "r") as f:
            return "\n".join(f.readlines())

    @staticmethod
    def dump_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class WebPageHelper:
    """Helper class to process web pages.

    Acknowledgement: Part of the code is adapted from https://github.com/stanford-oval/WikiChat project.
    """

    def __init__(
        self,
        min_char_count: int = 150,
        snippet_chunk_size: int = 500,
        snippet_chunk_overlap: int = 100,
        max_thread_num: int = 10,
    ):
        """
        Args:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            max_thread_num: Maximum number of threads to use for concurrent requests (e.g., downloading webpages).
        """
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=snippet_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )

    def download_webpage(self, url: str):
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            return res.content
        except httpx.HTTPError as exc:
            print(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None

    def urls_to_articles(self, urls: List[str]) -> Dict:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread_num
        ) as executor:
            lst_articles = list(executor.map(self.url_to_article, urls))
        articles = {}
        
        for article_text, u in zip(lst_articles, urls):
            
            # Ajouter le texte à 'articles' si sa longueur dépasse un seuil
            if article_text is not None and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}

        return articles

    def extract_pdf_text_from_url(self, pdf_url: str) -> str:
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()

            with fitz.open(stream=response.content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                    
            return text

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors du téléchargement : {e}")
        except Exception as e:
            print(f"Erreur lors de l'extraction du texte : {e}")

    def url_to_article(self, u: str) -> str:
        
        # print("Nombre de threads actifs :", threading.active_count())

        if "arxiv.org/pdf" in u:
            article_text = self.extract_pdf_text_from_url(u)
            return article_text
        else:
            # Télécharger la page HTML
            h = self.download_webpage(u)
            if h is not None :
                article_text = extract(
                    h,
                    include_tables=False,
                    include_comments=False,
                    output_format="txt",
                )
                return article_text
                
        


    def urls_to_snippets(self, urls: List[str]) -> Dict:
        articles = self.urls_to_articles(urls)
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"])

        return articles


class DemoTextProcessingHelper():

    @staticmethod
    def remove_citations(sent):
        return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

    @staticmethod
    def parse_conversation_history(json_data):
        """
        Given conversation log data, return list of parsed data of following format
        (persona_name, persona_description, list of dialogue turn)
        """
        parsed_data = []
        for persona_conversation_data in json_data:
            if ': ' in persona_conversation_data["perspective"]:
                name, description = persona_conversation_data["perspective"].split(": ", 1)
            elif '- ' in persona_conversation_data["perspective"]:
                name, description = persona_conversation_data["perspective"].split("- ", 1)
            else:
                name, description = "", persona_conversation_data["perspective"]
            cur_conversation = []
            for dialogue_turn in persona_conversation_data["dlg_turns"]:
                cur_conversation.append({"role": "user", "content": dialogue_turn["user_utterance"]})
                cur_conversation.append(
                    {"role": "assistant",
                     "content": DemoTextProcessingHelper.remove_citations(dialogue_turn["agent_utterance"])})
            parsed_data.append((name, description, cur_conversation))
        return parsed_data

    @staticmethod
    def parse(text):
        regex = re.compile(r']:\s+"(.*?)"\s+http')
        text = regex.sub(']: http', text)
        return text

    @staticmethod
    def add_markdown_indentation(input_string):
        lines = input_string.split('\n')
        processed_lines = [""]
        for line in lines:
            num_hashes = 0
            for char in line:
                if char == '#':
                    num_hashes += 1
                else:
                    break
            num_hashes -= 1
            num_spaces = 4 * num_hashes
            new_line = ' ' * num_spaces + line
            processed_lines.append(new_line)
        return '\n'.join(processed_lines)

    @staticmethod
    def get_current_time_string():
        """
        Returns the current time in the California time zone as a string.

        Returns:
            str: The current California time in 'YYYY-MM-DD HH:MM:SS' format.
        """
        california_tz = pytz.timezone('America/Los_Angeles')
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        california_now = utc_now.astimezone(california_tz)
        return california_now.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def compare_time_strings(time_string1, time_string2, time_format='%Y-%m-%d %H:%M:%S'):
        """
        Compares two time strings to determine if they represent the same point in time.

        Args:
            time_string1 (str): The first time string to compare.
            time_string2 (str): The second time string to compare.
            time_format (str): The format of the time strings, defaults to '%Y-%m-%d %H:%M:%S'.

        Returns:
            bool: True if the time strings represent the same time, False otherwise.
        """
        # Parse the time strings into datetime objects
        time1 = datetime.datetime.strptime(time_string1, time_format)
        time2 = datetime.datetime.strptime(time_string2, time_format)

        # Compare the datetime objects
        return time1 == time2

    @staticmethod
    def add_inline_citation_link(article_text, citation_dict):
        # Regular expression to find citations like [i]
        pattern = r'\[(\d+)\]'

        # Function to replace each citation with its Markdown link
        def replace_with_link(match):
            i = match.group(1)
            index_to_url = {v: k for k, v in citation_dict.items()}
            url = index_to_url.get(int(i), '#')
            if not url.startswith('http'):
                url = f"http://localhost:9000/{minioBucket}/{url}"
            return f'[[{i}]]({url})'
                

        # Replace all citations in the text with Markdown links
        return re.sub(pattern, replace_with_link, article_text)

    @staticmethod
    def generate_html_toc(md_text):
        toc = []
        for line in md_text.splitlines():
            if line.startswith("#"):
                level = line.count("#")
                title = line.strip("# ").strip()
                anchor = title.lower().replace(" ", "-").replace(".", "")
                toc.append(f"<li style='margin-left: {20 * (level - 1)}px;'><a href='#{anchor}'>{title}</a></li>")
        return "<ul>" + "".join(toc) + "</ul>"

    @staticmethod
    def construct_bibliography_from_url_to_info(url_to_info):
        bibliography_list = []
        sorted_url_to_unified_index = dict(sorted(url_to_info['url_to_unified_index'].items(),
                                                  key=lambda item: item[1]))
        for url, index in sorted_url_to_unified_index.items():
            if url_to_info['url_to_info'][url].title == "Titre non disponible":
                title = f"Lien pour {os.path.basename(url_to_info['url_to_info'][url].url)}"
                url = f"http://{minioHost}/{minioBucket}/{url}"
                url = os.path.join("result", url)
            else:
                title = url_to_info['url_to_info'][url].title
                title = title.replace("(", "").replace(")", "")
            bibliography_list.append(f"[{index}]: [{title}]({url})")
        bibliography_string = "\n\n".join(bibliography_list)
        return bibliography_string


def user_input_appropriateness_check(user_input):
    my_openai_model = OpenAIModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_provider="openai",
        model="gpt-4o-mini-2024-07-18",
        max_tokens=10,
        temperature=0.0,
        top_p=0.9,
    )

    if len(user_input.split()) > 20:
        return "The input is too long. Please make your input topic more concise!"

    if not re.match(r'^[a-zA-Z0-9\s\-"\,\.?\']*$', user_input):
        return "The input contains invalid characters. The input should only contain a-z, A-Z, 0-9, space, -/\"/,./?/'."

    prompt = f"""Here is a topic input into a knowledge curation engine that can write a Wikipedia-like article for the topic. Please judge whether it is appropriate or not for the engine to curate information for this topic based on English search engine. The following types of inputs are inappropriate:
1. Inputs that may be related to illegal, harmful, violent, racist, or sexual purposes.
2. Inputs that are given using languages other than English. Currently, the engine can only support English.
3. Inputs that are related to personal experience or personal information. Currently, the engine can only use information from the search engine.
4. Inputs that are not aimed at topic research or inquiry. For example, asks requiring detailed execution, such as calculations, programming, or specific service searches fall outside the engine's scope of capabilities.
If the topic is appropriate for the engine to process, output "Yes."; otherwise, output "No. The input violates reason [1/2/3/4]".
User input: {user_input}"""
    reject_reason_info = {
        1: "Sorry, this input may be related to sensitive topics. Please try another topic. "
        "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
        "We apologize for any inconvenience.)",
        2: "Sorry, the current engine can only support English. Please try another topic. "
        "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
        "We apologize for any inconvenience.)",
        3: "Sorry, the current engine cannot process topics related to personal experience. Please try another topic. "
        "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
        "We apologize for any inconvenience.)",
        4: "Sorry, STORM cannot follow arbitrary instruction. Please input a topic you want to learn about. "
        "(Our input filtering uses OpenAI GPT-4o-mini, which may result in false positives. "
        "We apologize for any inconvenience.)",
    }

    try:
        response = my_openai_model(prompt)[0].replace("[", "").replace("]", "")
        if response.startswith("No"):
            match = regex.search(r"reason\s(\d+)", response)
            if match:
                reject_reason = int(match.group(1))
                if reject_reason in reject_reason_info:
                    return reject_reason_info[reject_reason]
                else:
                    return (
                        "Sorry, the input is inappropriate. Please try another topic!"
                    )
            return "Sorry, the input is inappropriate. Please try another topic!"

    except Exception as e:
        return "Sorry, the input is inappropriate. Please try another topic!"
    return "Approved"


def purpose_appropriateness_check(user_input):
    my_openai_model = OpenAIModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_provider="openai",
        model="gpt-4o-mini-2024-07-18",
        max_tokens=10,
        temperature=0.0,
        top_p=0.9,
    )

    prompt = f"""
    Here is a purpose input into a report generation engine that can create a long-form report on any topic of interest. 
    Please judge whether the provided purpose is valid for using this service. 
    Try to judge if given purpose is non-sense like random words or just try to get around the sanity check.
    You should not make the rule too strict.
    
    If the purpose is valid, output "Yes."; otherwise, output "No" followed by reason.
    User input: {user_input}
    """
    try:
        response = my_openai_model(prompt)[0].replace("[", "").replace("]", "")
        if response.startswith("No"):
            return "Please provide a more detailed explanation on your purpose of requesting this article."

    except Exception as e:
        return "Please provide a more detailed explanation on your purpose of requesting this article."
    return "Approved"
    