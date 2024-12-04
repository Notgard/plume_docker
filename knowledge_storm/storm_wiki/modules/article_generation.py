import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing
from ...database import DataBase


class StormArticleGenerationModule_FR(ArticleGenerationModule):
    """
    L'interface pour la génération d'articles. Étant donné le sujet, les informations collectées
    lors de l'étape de knowledge curation et l'outline généré lors de l'étape de outline generation,
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection_FR(engine=self.article_gen_lm)
        
        self.apply_decorators()

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query
    ):
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        return {
            "nom_section": section_name,
            "contenu_section": output.section,
            "info_collectee": collected_info,
        }

    def generate_article(
        self,
        topic: str,
        database: DataBase,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Générer un article pour le sujet basé sur la table d'informations et l'outline de l'article.

            Args:
                topic (str): Le sujet de l'article.
                information_table (StormInformationTable): La table d'informations contenant les informations collectées.
                article_with_outline (StormArticle): L'article avec l'outline spécifié.
                callback_handler (BaseCallbackHandler): Un gestionnaire de rappel facultatif qui peut être utilisé pour déclencher
                    des rappels personnalisés à différentes étapes du processus de génération d'articles. Par défaut, None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        sections_to_write = article_with_outline.get_first_level_section_names()
        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(
                f"Pas d'outline pour {topic}. Recherche directe sur le sujet."
            )
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic],
            )
            section_output_dict_collection = [section_output_dict]
        else:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    status = f"Génération de la section {section_title} ..."
                    database.update_status_topic(topic, status)
                    # We don't want to write a separate introduction section.
                    if section_title.lower().strip() == "introduction":
                        continue
                        # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section,
                            topic,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())
        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                section_name=section_output_dict["nom_section"],
                current_section_content=section_output_dict["contenu_section"],
                current_section_info_list=section_output_dict["info_collectee"],
            )
        article.post_processing()
        return article


class ConvToSection_FR(dspy.Module):
    """Utilisez les informations collectées lors de la conversation d'information pour rédiger une section en Français."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection_FR)
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information]
    ):
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"

        # info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1500)
        print (topic)
        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output
            )
        print(f"Résultat: {section}")
        return dspy.Prediction(section=section)


class WriteSection_FR(dspy.Signature):
    """Rédigez une section Wikipédia basée sur les informations collectées.

   Voici le format de votre rédaction :
        1. Utilisez "#" Titre" pour indiquer le titre de la section, "##" Titre" pour indiquer le titre de la sous-section, "###" Titre" pour indiquer le titre de la sous-sous-section, et ainsi de suite.
        2. Utilisez [1], [2], ..., [n] dans le texte (par exemple, "La capitale des États-Unis est Washington, D.C.[1][3]."). NE PAS inclure une section Références ou Sources pour lister les sources à la fin.
    """

    info = dspy.InputField(prefix="Les informations collectées :\n", format=str)
    topic = dspy.InputField(prefix="Le sujet de la page :", format=str)
    section = dspy.InputField(prefix="La section que tu dois écrire, en Français :", format=str)
    output = dspy.OutputField(
        prefix="Rédigez la section en Français avec les citations en ligne appropriées (commencez votre rédaction par le titre de la section #. N'incluez pas le titre de la page ou n'essayez pas d'écrire d'autres sections), ne pas mettre Références ou Sources à la fin. Utilise [1], [2], ..., [n] au sein du texte pour ajouter les références dans la section:\n",
        format=str,
    )


class StormArticleGenerationModule_EN(ArticleGenerationModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage,
    """

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection_EN(engine=self.article_gen_lm)

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query
    ):
        collected_info: List[Information] = []
        if information_table is not None:
            collected_info = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_article(
        self,
        topic: str,
        database: DataBase,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Generate article for the topic based on the information table and article outline.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            article_with_outline (StormArticle): The article with specified outline.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the article generation process. Defaults to None.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        sections_to_write = article_with_outline.get_first_level_section_names()

        section_output_dict_collection = []
        if len(sections_to_write) == 0:
            logging.error(
                f"No outline for {topic}. Will directly search with the topic."
            )
            section_output_dict = self.generate_section(
                topic=topic,
                section_name=topic,
                information_table=information_table,
                section_outline="",
                section_query=[topic],
            )
            section_output_dict_collection = [section_output_dict]
        else:

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                future_to_sec_title = {}
                for section_title in sections_to_write:
                    status = f"Génération de la section {section_title}..."
                    database.update_status_topic(topic, status)
                    # We don't want to write a separate introduction section.
                    if section_title.lower().strip() == "introduction":
                        continue
                        # We don't want to write a separate conclusion section.
                    if section_title.lower().strip().startswith(
                        "conclusion"
                    ) or section_title.lower().strip().startswith("summary"):
                        continue
                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    queries_with_hashtags = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=True
                    )
                    section_outline = "\n".join(queries_with_hashtags)
                    future_to_sec_title[
                        executor.submit(
                            self.generate_section,
                            topic,
                            section_title,
                            information_table,
                            section_outline,
                            section_query,
                        )
                    ] = section_title

                for future in as_completed(future_to_sec_title):
                    section_output_dict_collection.append(future.result())
        article = copy.deepcopy(article_with_outline)
        for section_output_dict in section_output_dict_collection:
            article.update_section(
                section_name=section_output_dict["section_name"],
                current_section_content=section_output_dict["section_content"],
                current_section_info_list=section_output_dict["collected_info"],
            )
        article.post_processing()
        return article


class ConvToSection_EN(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection_EN)
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information]
    ):
        info = ""
        for idx, storm_info in enumerate(collected_info):
            info += f"[{idx + 1}]\n" + "\n".join(storm_info.snippets)
            info += "\n\n"
        
        # print("longuer d'info", len(info))

        with dspy.settings.context(lm=self.engine):
            section = ArticleTextProcessing.clean_up_section(
                self.write_section(topic=topic, info=info, section=section).output
            )

        return dspy.Prediction(section=section)


class WriteSection_EN(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). DO NOT include a References or Sources section to list the sources at the end.
    """
    info = dspy.InputField(prefix="The collected information in English:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page in English:", format=str)
    section = dspy.InputField(prefix="The section you need to write in English:", format=str)
    output = dspy.OutputField(
        prefix="Write the section in English with appropriate inline citations (start your writing with the section title #. Do not include the title of the page or attempt to write other sections) and don't put a section References or Sources. Use [1], [2], ..., [n] in line to include references inside the section:\n",
        format=str,
    )