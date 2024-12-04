import copy
from typing import Union
import dspy
import re

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing

class StormArticlePolishingModule_FR(ArticlePolishingModule):
    """
    L'interface pour l'étape de génération d'articles.
    A partir d'un sujet donné, des informations sont collectées à partir de l'étape de knowledge curation,
    et une outline est générée à partir de l'étape de outline generation.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm
        self.polish_page = PolishPageModule_FR(
            write_lead_engine=self.article_gen_lm,
            polish_engine=self.article_polish_lm
        )
        
        self.apply_decorators()


    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polir l'article.

        Args:
            topic (str): Le sujet de l'article.
            draft_article (StormArticle): L'article brouillon.
            remove_duplicate (bool): Utiliser un appel LM supplémentaire pour supprimer les doublons de l'article.
        """

        # Polish article
        article_text = draft_article.to_string()
        polish_result = self.polish_page(
            topic=topic,
            draft_page=article_text,
            polish_whole_page=remove_duplicate
        )
        
        lead_section = f"# Résumé\n{polish_result.lead_section}"
        polished_article = "\n\n".join([lead_section, polish_result.page])
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )
        
        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()
        polished_article.link_citation()
        polished_article.add_reference()
        return polished_article


class WriteLeadSection_FR(dspy.Signature):
    """Rédigez une section principale pour la page Wikipédia donnée en suivant les directives suivantes :
    1. L'introduction doit être un aperçu concis du sujet de l'article. Il doit identifier le sujet, établir le contexte, expliquer pourquoi le sujet est important et résumer les points les plus importants, y compris toute controverse importante.
    2. La section principale (lead section) doit être concise et ne pas contenir plus de quatre paragraphes bien composés.
    3. La section principale (lead section) ne doit pas être sourcée, le cas échéant. Supprimer les citations en ligne (par exemple, « Washington, D.C., est la capitale des États-Unis[1][3]. ») si nécessaire.
    """
    topic = dspy.InputField(prefix="Le sujet de la page en Français :", format=str)
    draft_page = dspy.InputField(prefix="La page de brouillon en Français :\n", format=str)
    lead_section = dspy.OutputField(
        prefix="Rédiger la section principale en Français :\n", 
        format=str
    )



class PolishPage_FR(dspy.Signature):
    """Vous êtes un éditeur de texte fidèle qui sait trouver les informations répétées
    dans l'article et les supprimer pour s'assurer qu'il n'y a pas de répétition dans l'article.
    Vous ne supprimerez aucune partie non répétée de l'article. Vous conserverez les citations en ligne
    et la structure de l'article (indiquée par « # », « ## », etc.) de manière appropriée.
    Faites votre travail pour l'article suivant.
    """
    draft_page = dspy.InputField(prefix="L'article brouillon en Français :\n", format=str)
    page = dspy.OutputField(prefix="Votre article révisé en Français :\n", format=str)


class PolishPageModule_FR(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection_FR)
        self.polish_page = dspy.Predict(PolishPage_FR)
    

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # Set the language context
        print(f"Langue utilisée : Français")
        
        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = ArticleTextProcessing.clean_up_section(
                self.write_lead(
                    topic=topic, draft_page=draft_page
                    ).lead_section
            )
            
            # Remove unwanted phrases
        lead_section = re.sub(r'.*section principale[^:]*:', '', lead_section)
        lead_section = ArticleTextProcessing.limit_word_count_preserve_newline(lead_section, 1500)

        if polish_whole_page:
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            page = draft_page

        return dspy.Prediction(lead_section=lead_section, page=page)
    


class StormArticlePolishingModule_EN(ArticlePolishingModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm
        self.polish_page = PolishPageModule_EN(
            write_lead_engine=self.article_gen_lm,
            polish_engine=self.article_polish_lm
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polish article.

        Args:
            topic (str): The topic of the article.
            draft_article (StormArticle): The draft article.
            remove_duplicate (bool): Whether to use one additional LM call to remove duplicates from the article.
        """

        # Polish article
        article_text = draft_article.to_string()
        polish_result = self.polish_page(
            topic=topic,
            draft_page=article_text,
            polish_whole_page=remove_duplicate
        )
        
        lead_section = f"# Summary\n{polish_result.lead_section}"
        polished_article = "\n\n".join([lead_section, polish_result.page])
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )
        
        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()
        polished_article.link_citation()
        polished_article.add_reference()
        return polished_article


class WriteLeadSection_EN(dspy.Signature):
    """Write a lead section for the given Wikipedia page with the following guidelines:
    1. The lead should stand on its own as a concise overview of the article's topic. It should identify the topic, establish context, explain why the topic is notable, and summarize the most important points, including any prominent controversies.
    2. The lead section should be concise and contain no more than four well-composed paragraphs.
    3. The lead section should not be sourced. Remove inline citations (e.g., "Washington, D.C., is the capital of the United States.[1][3].") where necessary.
    """

    topic = dspy.InputField(prefix="The topic of the page in English:", format=str)
    draft_page = dspy.InputField(prefix="The draft page in English:\n", format=str)
    lead_section = dspy.OutputField(
        prefix="Write the lead section in English:\n", 
        format=str
    )



class PolishPage_EN(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article."""

    draft_page = dspy.InputField(prefix="The draft article in English:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article in English:\n", format=str)


class PolishPageModule_EN(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection_EN)
        self.polish_page = dspy.Predict(PolishPage_EN)
    

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # Set the language context
        print(f"Language used for generation: English")

        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = ArticleTextProcessing.clean_up_section(
                self.write_lead(
                    topic=topic, draft_page=draft_page
                    ).lead_section
                )
            
            # Remove unwanted phrases
        lead_section = re.sub(r'.*lead section[^:]*:', '', lead_section)
        lead_section = ArticleTextProcessing.limit_word_count_preserve_newline(lead_section, 1500)

        if polish_whole_page:
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            page = draft_page

        return dspy.Prediction(lead_section=lead_section, page=page)