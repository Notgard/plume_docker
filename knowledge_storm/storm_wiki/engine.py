import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy
from langdetect import detect
import requests
from dotenv import dotenv_values
from minio.error import S3Error
from io import BytesIO
from sqlalchemy import desc

from .modules.article_generation import StormArticleGenerationModule_FR, StormArticleGenerationModule_EN
from .modules.article_polish import StormArticlePolishingModule_FR, StormArticlePolishingModule_EN
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule_EN, StormOutlineGenerationModule_FR
from .modules.persona_generator import StormPersonaGenerator
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs, Retriever
from ..lm import OpenAIModel, AzureOpenAIModel
from ..utils import FileIOHelper, makeStringRed, truncate_filename
from ..database import DataBase, JSONDocument, Topic, MDDocument

config = {**dotenv_values(".env"),
          **os.environ,}
minioBucket = config["MINIO_BUCKET_NAME"]

class STORMWikiLMConfigs(LMConfigs):
    """Configurations for LLM used in different parts of STORM.

    Given that different parts in STORM framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        self.conv_simulator_lm = (
            None  # LLM used in conversation simulator except for question asking.
        )
        self.question_asker_lm = None  # LLM used in question asking.
        self.outline_gen_lm = None  # LLM used in outline generation.
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.

    def init_openai_model(
        self,
        openai_api_key: str,
        azure_api_key: str,
        openai_type: Literal["openai", "azure"],
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Legacy: Corresponding to the original setup in the NAACL'24 paper."""
        azure_kwargs = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }

        openai_kwargs = {
            "api_key": openai_api_key,
            "api_provider": "openai",
            "temperature": temperature,
            "top_p": top_p,
            "api_base": None,
        }
        if openai_type and openai_type == "openai":
            self.conv_simulator_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            # 1/12/2024: Update gpt-4 to gpt-4-1106-preview. (Currently keep the original setup when using azure.)
            self.outline_gen_lm = OpenAIModel(
                model="gpt-4-0125-preview", max_tokens=400, **openai_kwargs
            )
            self.article_gen_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=700, **openai_kwargs
            )
            self.article_polish_lm = OpenAIModel(
                model="gpt-4o-2024-05-13", max_tokens=4000, **openai_kwargs
            )
        elif openai_type and openai_type == "azure":
            self.conv_simulator_lm = OpenAIModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **openai_kwargs
            )
            self.question_asker_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=500,
                **azure_kwargs,
                model_type="chat",
            )
            # use combination of openai and azure-openai as azure-openai does not support gpt-4 in standard deployment
            self.outline_gen_lm = AzureOpenAIModel(
                model="gpt-4o", max_tokens=400, **azure_kwargs, model_type="chat"
            )
            self.article_gen_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=700,
                **azure_kwargs,
                model_type="chat",
            )
            self.article_polish_lm = AzureOpenAIModel(
                model="gpt-4o-mini-2024-07-18",
                max_tokens=4000,
                **azure_kwargs,
                model_type="chat",
            )
        else:
            logging.warning(
                "No valid OpenAI API provider is provided. Cannot use default LLM configurations."
            )

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""

    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={
            "help": "Maximum number of questions in conversational question asking."
        },
    )
    max_perspective: int = field(
        default=3,
        metadata={
            "help": "Maximum number of perspectives to consider in perspective-guided question asking."
        },
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )


class STORMWikiRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(
        self, args: STORMWikiRunnerArguments, lm_configs: STORMWikiLMConfigs, rm
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)
        storm_persona_generator = StormPersonaGenerator(
            self.lm_configs.question_asker_lm
        )
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=storm_persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )


    def detect_language(self, text: str) -> str:
        """Detects the language of the text (e.g., the topic)."""
        return detect(text)

    selected_language = None
    def set_selected_language(language):
        global selected_language
        selected_language = language

    def init_language_modules(self, nbr_paragraphe):
        """Initialize language-specific modules based on the topic's language."""
        global selected_language
        print(f"Langue sélectionnée : {selected_language}")

        if selected_language:
            language = selected_language
        else:
            language = self.detect_language(self.topic)

        print(f"Langue utilisée : {language}")
        if language == "fr":
            print("Language used for generation: French")
            self.storm_outline_generation_module = StormOutlineGenerationModule_FR(
                outline_gen_lm=self.lm_configs.outline_gen_lm,
                nbr_paragraphe=nbr_paragraphe
            )
            self.storm_article_generation = StormArticleGenerationModule_FR(
                article_gen_lm=self.lm_configs.article_gen_lm,
                retrieve_top_k=self.args.retrieve_top_k,
                max_thread_num=self.args.max_thread_num,
            )
            self.storm_article_polishing_module = StormArticlePolishingModule_FR(
                article_gen_lm=self.lm_configs.article_gen_lm,
                article_polish_lm=self.lm_configs.article_polish_lm,
            )
        else:
            if language == "en":
                print("Language used for generation: English")
            else:
                print(f"Unsupported language detected: {language}")
                logging.warning(f"Unsupported language detected: {language}")
            
            self.storm_outline_generation_module = StormOutlineGenerationModule_EN(
                outline_gen_lm=self.lm_configs.outline_gen_lm,
                nbr_paragraphe=nbr_paragraphe
            )
            self.storm_article_generation = StormArticleGenerationModule_EN(
                article_gen_lm=self.lm_configs.article_gen_lm,
                retrieve_top_k=self.args.retrieve_top_k,
                max_thread_num=self.args.max_thread_num,
            )
            self.storm_article_polishing_module = StormArticlePolishingModule_EN(
                article_gen_lm=self.lm_configs.article_gen_lm,
                article_polish_lm=self.lm_configs.article_polish_lm,
            )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        database: DataBase,
        client,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None
    ) -> StormInformationTable:

        information_table, conversation_log = (
            self.storm_knowledge_curation_module.research(
                topic=self.topic,
                database=database,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
                max_perspective=self.args.max_perspective,
                disable_perspective=False,
                return_conversation_log=True,
            )
        )

        json_data = json.dumps(conversation_log)
        json_bytes = BytesIO(json_data.encode("utf-8"))
        
        logs_path = os.path.join(self.article_output_dir, "conversation_log.json")

        try:
            client.put_object(minioBucket, logs_path, data=json_bytes, length = len(json_data), content_type = "application/json")
        except S3Error as e:
            print(f"Error while creating 'conversation_log.json' : {e}")
        session = database.create_session()
        latest_topic = (
            session.query(Topic)
            .filter_by(topic=self.topic)
            .order_by(desc(Topic.start_date))
            .first()
        )

        json_conversation = JSONDocument(
            object_link = logs_path,
            topic_id = latest_topic.id
        )

        session.add(json_conversation)
        session.commit()
        session.refresh(json_conversation)
        session.close()
        
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json"), client = client, database = database, topic=self.topic
        )
        return information_table

    def run_outline_generation_module(
        self,
        information_table: StormInformationTable,
        client, 
        database,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:

        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler,
        )
        outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "storm_gen_outline.md"), client=client, database=database, topic = self.topic
        )
        draft_outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "direct_gen_outline.md"), client=client, database=database, topic = self.topic
        )
        return outline

    def run_article_generation_module(
        self,
        database: DataBase,
        outline: StormArticle,
        client,
        information_table=StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
    
        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            database=database,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler,
        )
        draft_article.dump_article_as_plain_text(
            os.path.join(self.article_output_dir, "storm_gen_article.md"), client = client, database = database, topic = self.topic
        )
        draft_article.dump_reference_to_file(
            os.path.join(self.article_output_dir, "url_to_info.json"), client = client, database = database, topic = self.topic
        )
        return draft_article

    def run_article_polishing_module(
        self, database: DataBase, draft_article: StormArticle, client, remove_duplicate: bool = False
    ) -> StormArticle:

        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,
        )

        file_path = os.path.join(self.article_output_dir, "storm_gen_article_polished.md")
        text = polished_article.to_string()
        markdown = BytesIO(text.encode("utf-8"))

        try:
            client.put_object(minioBucket, file_path, data=markdown, length = len(text), content_type = "application/json")
        except S3Error as e:
            print(f"Error while creating 'storm_gen_article_polished.md' : {e}")
        session = database.create_session()
        latest_topic = (
            session.query(Topic)
            .filter_by(topic=self.topic)
            .order_by(desc(Topic.start_date))
            .first()
        )

        article_polished = MDDocument(
            object_link = file_path,
            topic_id = latest_topic.id
        )

        session.add(article_polished)
        session.commit()
        session.refresh(article_polished)
        session.close()
        return polished_article

    def post_run(self, client, database):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.lm_configs.log()
        file_path = os.path.join(self.article_output_dir, "run_config.json")
        
        json_data = json.dumps(config_log)
        json_bytes = BytesIO(json_data.encode("utf-8"))

        try:
            client.put_object(minioBucket, file_path, data=json_bytes, length = len(json_data), content_type = "application/json")
        except S3Error as e:
            print(f"Error while creating 'run_config.json' : {e}")
        session = database.create_session()
        latest_topic = (
            session.query(Topic)
            .filter_by(topic=self.topic)
            .order_by(desc(Topic.start_date))
            .first()
        )

        json_conversation = JSONDocument(
            object_link = file_path,
            topic_id = latest_topic.id
        )

        session.add(json_conversation)
        session.commit()
        session.refresh(json_conversation)
        session.close()
        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        jsonl_content = ""
        file_path = os.path.join(self.article_output_dir, "llm_call_history.jsonl")
       
        for call in llm_call_history:
            if "kwargs" in call:
                call.pop(
                    "kwargs"
                )  # All kwargs are dumped together to run_config.json.
            for key, value in call.items() :
                if hasattr(value, 'to_dict'):
                    call[key]=value.to_dict()
            jsonl_content += json.dumps(call) + "\n"

        jsonl_bytes = BytesIO(jsonl_content.encode('utf-8'))

        try:
            client.put_object(
                minioBucket, file_path, data = jsonl_bytes, length = len(jsonl_content.encode('utf-8')), content_type="application/jsonl"
            )
        except S3Error as e:
            print(f"Error while creating llm_call_history.jsonl file: {e}")
            
        session = database.create_session()
        latest_topic = (
            session.query(Topic)
            .filter_by(topic=self.topic)
            .order_by(desc(Topic.start_date))
            .first()
        )

        llm_call_history = JSONDocument(
            object_link = file_path,
            topic_id = latest_topic.id
        )

        session.add(llm_call_history)
        session.commit()
        session.refresh(llm_call_history)
        session.close()

    def _load_information_table_from_local_fs(self, information_table_local_path, client):
        try:
            client.stat_object(minioBucket, information_table_local_path)
            return StormInformationTable.from_conversation_log_file(
                information_table_local_path, client
            )
        except S3Error as e:
            if e.code == "NoSuchKey":
                makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-generate-outline argument to prepare the storm_gen_outline.md for this topic."
                )
            else:
                print(f"Error while verifying the existence of infomration table: {e}")


    def _load_outline_from_local_fs(self, topic, outline_local_path, client):
        try:
            client.stat_object(minioBucket, outline_local_path)
            return StormArticle.from_outline_file(topic=topic, file_path=outline_local_path, client=client)
        except S3Error as e:
            if e.code == "NoSuchKey":
                makeStringRed(
            f"{outline_local_path} not exists. Please set --do-generate-outline argument to prepare the storm_gen_outline.md for this topic."
        )
            else:
                print(f"Error while verifying the existence of outline: {e}")        

    def _load_draft_article_from_local_fs(
        self, topic, draft_article_path, url_to_info_path, client
    ):
        try:
            article_text_response = client.get_object(minioBucket, draft_article_path)
            try:
                references_response = client.get_object(minioBucket, url_to_info_path)
                article_text = article_text_response.read().decode('utf-8')
                references = json.load(references_response)
                references_response.close()
                references_response.release_conn()
                article_text_response.close()
                article_text_response.release_conn()
        
                return StormArticle.from_string(
                    topic_name=topic, article_text=article_text, references=references
                )
            except S3Error as e:
                print(f"Error loading the references: {e}")
        except S3Error as e:
            print(f"Error loading the article text: {e}")

        article_text = FileIOHelper.load_str(draft_article_path)
        references = FileIOHelper.load_json(url_to_info_path)

        try:
            response = client.get_object(minioBucket, draft_article_path)
            article_text = response.read().decode('utf-8')
            response.close()
            response.release_conn()
        except S3Error as e:
            print(f"Error while reading the file: {e}")

        try:
            response = client.get_object(minioBucket, draft_article_path)
            reference_content = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            references = json.loads(reference_content)
        except S3Error as e:
            print(f"Error while reading the file: {e}")
        
        return StormArticle.from_string(
            topic_name=topic, article_text=article_text, references=references
        )

    def run(
        self,
        topic: str,
        database: DataBase,
        nbr_paragraphe: str,
        client,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler()
    ):
        """
        Run the STORM pipeline.

        Args:
            topic: The topic to research.
            ground_truth_url: A ground truth URL including a curated article about the topic. The URL will be excluded.
            do_research: If True, research the topic through information-seeking conversation;
             if False, expect conversation_log.json and raw_search_results.json to exist in the output directory.
            do_generate_outline: If True, generate an outline for the topic;
             if False, expect storm_gen_outline.md to exist in the output directory.
            do_generate_article: If True, generate a curated article for the topic;
             if False, expect storm_gen_article.md to exist in the output directory.
            do_polish_article: If True, polish the article by adding a summarization section and (optionally) removing
             duplicated content.
            remove_duplicate: If True, remove duplicated content.
            callback_handler: A callback handler to handle the intermediate results.
        """
        assert (
            do_research
            or do_generate_outline
            or do_generate_article
            or do_polish_article
        ), makeStringRed(
            "No action is specified. Please set at least one of --do-research, --do-generate-outline, --do-generate-article, --do-polish-article"
        )

        self.topic = topic

        self.article_output_dir = os.path.join(
            self.args.output_dir, "productions"
        )

        self.init_language_modules(nbr_paragraphe)
        
        # research module
        information_table: StormInformationTable = None
        if do_research:
            database.update_status_topic(topic, "Lancement du premier module: Knowledge Curation...")
            information_table = self.run_knowledge_curation_module(
                database=database,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
                client=client
            )
            database.update_status_topic(topic, "Premier module terminé.")
        # outline generation module
        outline: StormArticle = None
        if do_generate_outline:
            database.update_status_topic(topic, "Lancement du deuxième Module: Outline Generation...")
            # load information table if it's not initialized
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json"), client
                )
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler, client = client, database = database
            )
            database.update_status_topic(topic, "Deuxième module terminé.")

        # article generation module
        draft_article: StormArticle = None
        if do_generate_article:
            database.update_status_topic(topic, "Lancement du troisième module : Article Generation...")
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            if outline is None:
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    outline_local_path=os.path.join(
                        self.article_output_dir, "storm_gen_outline.md"
                    ),
                )
            draft_article = self.run_article_generation_module(
                database=database,
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
                client = client
            )
            database.update_status_topic(topic, "Troisème module terminé.")

        # article polishing module
        if do_polish_article:
            database.update_status_topic(topic, "Lancement du quatrième module : Article Polish...")
            if draft_article is None:
                draft_article_path = os.path.join(
                    self.article_output_dir, "storm_gen_article.md"
                )
                url_to_info_path = os.path.join(
                    self.article_output_dir, "url_to_info.json"
                )
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_article_path=draft_article_path,
                    url_to_info_path=url_to_info_path,
                    client=client
                )
            self.run_article_polishing_module(
                database=database, draft_article=draft_article, remove_duplicate=remove_duplicate, client = client
            )
            database.update_status_topic(topic, "Quatrième module terminé.")
            database.update_status_topic(topic, "Redirection vers la page de résultats...")