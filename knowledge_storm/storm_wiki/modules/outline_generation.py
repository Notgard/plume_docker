from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


class StormOutlineGenerationModule_FR(OutlineGenerationModule):
    """
    L'interface pour l'étape de génération des grandes lignes. A partir d'un sujet donné et d'informations collectées
    l'étape de curation des connaissances, générer le plan de l'article.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel], nbr_paragraphe: str):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.nbr_paragraphe = nbr_paragraphe
        self.write_outline = WriteOutline_FR(engine=self.outline_gen_lm, nbr_paragraphe=self.nbr_paragraphe)
        
        self.apply_decorators()

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline=False,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        Génère un plan (outline) d'article basé sur le sujet spécifié et les informations recueillies lors de l'étape de
        recueillies au cours de l'étape de curation des connaissances. Cette méthode peut optionnellement renvoyer à la fois le
        l'ébauche finale de l'article et une ébauche de l'ébauche si nécessaire.
        Args :
            topic (str) : Le sujet de l'article.
            information_table (StormInformationTable) : La table d'information contenant les informations collectées.
            old_outline (Optional[StormArticle]) : Une version antérieure facultative du plan de l'article qui peut
                être utilisée à des fins de référence ou de comparaison. La valeur par défaut est None.
            callback_handler (BaseCallbackHandler) : Un gestionnaire de rappel facultatif qui peut être utilisé pour déclencher des rappels personnalisés à différents stades de l'article.
                déclencher des rappels personnalisés à différents stades du processus de génération des grandes lignes, par exemple lorsque l'organisation des informations commence.
                l'organisation de l'information. La valeur par défaut est None.
            return_draft_outline (bool) : Indicateur indiquant si la méthode doit renvoyer à la fois le plan final de l'article et une version préliminaire du plan.
                finale de l'article et une version préliminaire du plan. Si False, seule la version finale de l'article est renvoyée.
                La valeur par défaut est False.
        Retourne :
            Union[StormArticle, Tuple[StormArticle, StormArticle]] : En fonction de la valeur de `return_draft_outline`,
                cette méthode renvoie soit un seul objet `StormArticle` contenant le plan final, soit un tuple de deux `StormArticle`.
                deux objets `StormArticle`, le premier contenant le contour final et le second, le
        """
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        concatenated_dialogue_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )
        result = self.write_outline(
            topic=topic,
            dlg_history=concatenated_dialogue_turns,
            callback_handler=callback_handler,
        )
        article_with_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.outline
        )
        article_with_draft_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.old_outline
        )
        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline_FR(dspy.Module):
    """Générer le plan (outline) de la page Wikipédia."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], nbr_paragraphe: str):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline_FR)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv_FR)
        self.engine = engine
        self.nbr_paragraphe = nbr_paragraphe

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        trimmed_dlg_history = []
        for turn in dlg_history:
            if (
                "topic you" in turn.agent_utterance.lower()
                or "topic you" in turn.user_utterance.lower()
            ):
                continue
            trimmed_dlg_history.append(turn)
        conv = "\n".join(
            [
                f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
                for turn in trimmed_dlg_history
            ]
        )
        conv = ArticleTextProcessing.remove_citations(conv)
        # conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 5000)

        with dspy.settings.context(lm=self.engine):
            if old_outline is None:
                old_outline = ArticleTextProcessing.clean_up_outline(
                    self.draft_page_outline(topic=topic).outline
                )
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(
                        outline=old_outline
                    )
            num_sections = self.nbr_paragraphe
            outline = ArticleTextProcessing.clean_up_outline(
                self.write_page_outline(
                    topic=topic, old_outline=old_outline, conv=conv, num_sections=num_sections
                ).outline
            # With Ollama the new outline is badly influenced by the old
            )
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)

        return dspy.Prediction(outline=outline, old_outline=old_outline)


class WritePageOutline_FR(dspy.Signature):
    """Rédigez le plan d'une page Wikipédia.
    Voici le format de votre rédaction :
    1. Utilisez "#" Titre"  pour indiquer le titre de la section, "##" Titre" pour indiquer le titre de la sous-section, "###" Titre" pour indiquer le titre de la sous-section, et ainsi de suite.
    2. Ne pas inclure d'autres informations.
    3. Ne pas inclure le nom du sujet lui-même dans le plan.
    4. **Le plan doit être rédigé en français.**
    """

    topic = dspy.InputField(prefix="Le sujet sur lequel vous souhaitez écrire : ", format=str)
    outline = dspy.OutputField(prefix='Rédiger le plan de la page Wikipédia sur la base de la conversation en français.\n', format=str)


class NaiveOutlineGen_FR(dspy.Module):
    """Générer le contour avec la connaissance paramétrique de LLM directement."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline_FR)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline

        return dspy.Prediction(outline=outline)


class WritePageOutlineFromConv_FR(dspy.Signature):
    """Améliorer le plan d'une page Wikipédia. Vous avez déjà une ébauche de plan qui couvre les informations générales. Vous voulez maintenant l'améliorer en vous basant sur les informations apprises lors d'une conversation de recherche d'informations afin de la rendre plus informative.
    Voici le format de votre texte :
    1. Utilisez "#" Titre" pour indiquer le titre de la section, “##” Titre" pour indiquer le titre de la sous-section, "###" Titre" pour indiquer le titre de la sous-section, et ainsi de suite.
    2. Ne pas inclure d'informations ou d'explications supplémentaires en dehors des titres.
    3. Ne pas inclure le nom du sujet lui-même dans les titres du plan.
    4. Le plan doit contenir exactement {num_sections} sections, ni plus ni moins.
    5. **Le plan doit être rédigé en français.**
    """
    topic = dspy.InputField(prefix="Le sujet sur lequel vous souhaitez écrire : ", format=str)
    conv = dspy.InputField(prefix="L'historique de la conversation :\n", format=str)
    old_outline = dspy.InputField(prefix="Le plan (outline) actuel :\n", format=str)
    num_sections = dspy.InputField(prefix="Le nombre de section(s) :\n", format=str)
    outline = dspy.OutputField(
        prefix=(
            'Rédiger le plan de la page Wikipédia sur la base de la conversation en français.\n'
            'Rappellez-vous :\n'
            '- Utilisez "#" Titre" pour les sections, "##" Titre" pour les sous-sections, etc.\n'
            '- Ne pas inclure le sujet dans les titres.\n'
            '- Inclure exactement {num_sections} sections.\n'
            'Voici le plan amélioré en français :\n'
        ),
        format=str,
    )



class StormOutlineGenerationModule_EN(OutlineGenerationModule):
    """
    The interface for outline generation stage. Given topic, collected information from knowledge
    curation stage, generate outline for the article.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel], nbr_paragraphe: str):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.nbr_paragraphe = nbr_paragraphe
        self.write_outline = WriteOutline_EN(engine=self.outline_gen_lm, nbr_paragraphe=self.nbr_paragraphe)
        
        self.apply_decorators()

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline=False,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        Generates an outline for an article based on the specified topic and the information
        gathered during the knowledge curation stage. This method can optionally return both the
        final article outline and a draft outline if required.

        Args:
            topic (str): The topic of the article.
            information_table (StormInformationTable): The information table containing the collected information.
            old_outline (Optional[StormArticle]): An optional previous version of the article outline that can
                be used for reference or comparison. Defaults to None.
            callback_handler (BaseCallbackHandler): An optional callback handler that can be used to trigger
                custom callbacks at various stages of the outline generation process, such as when the information
                organization starts. Defaults to None.
            return_draft_outline (bool): A flag indicating whether the method should return both the final article
                outline and a draft version of the outline. If False, only the final article outline is returned.
                Defaults to False.

        Returns:
            Union[StormArticle, Tuple[StormArticle, StormArticle]]: Depending on the value of `return_draft_outline`,
                this method returns either a single `StormArticle` object containing the final outline or a tuple of
                two  `StormArticle` objects, the first containing the final outline and the second containing the
                draft outline.
        """
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        concatenated_dialogue_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )
        result = self.write_outline(
            topic=topic,
            dlg_history=concatenated_dialogue_turns,
            callback_handler=callback_handler,
        )
        article_with_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.outline
        )
        article_with_draft_outline_only = StormArticle.from_outline_str(
            topic=topic, outline_str=result.old_outline
        )
        if not return_draft_outline:
            return article_with_outline_only
        return article_with_outline_only, article_with_draft_outline_only


class WriteOutline_EN(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], nbr_paragraphe: str):
        super().__init__()
        self.draft_page_outline = dspy.Predict(WritePageOutline_EN)
        self.write_page_outline = dspy.Predict(WritePageOutlineFromConv_EN)
        self.engine = engine
        self.nbr_paragraphe = nbr_paragraphe

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        trimmed_dlg_history = []
        for turn in dlg_history:
            if (
                "topic you" in turn.agent_utterance.lower()
                or "topic you" in turn.user_utterance.lower()
            ):
                continue
            trimmed_dlg_history.append(turn)
        conv = "\n".join(
            [
                f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
                for turn in trimmed_dlg_history
            ]
        )
        conv = ArticleTextProcessing.remove_citations(conv)

        with dspy.settings.context(lm=self.engine):
            if old_outline is None:
                old_outline = ArticleTextProcessing.clean_up_outline(
                    self.draft_page_outline(topic=topic).outline
                )
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(
                        outline=old_outline
                    )
            num_sections = self.nbr_paragraphe
            outline = ArticleTextProcessing.clean_up_outline(
                self.write_page_outline(
                    topic=topic, old_outline=old_outline, conv=conv, num_sections=num_sections
                ).outline
            )
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=outline)

        return dspy.Prediction(outline=outline, old_outline=old_outline)


class WritePageOutline_EN(dspy.Signature):
    """Write an outline for a Wikipedia page.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include other information.
    3. Do not include topic name itself in the outline.
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    outline = dspy.OutputField(prefix="Write the Wikipedia page outline:\n", format=str)


class NaiveOutlineGen_EN(dspy.Module):
    """Generate the outline with LLM's parametric knowledge directly."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline_EN)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline

        return dspy.Prediction(outline=outline)

class WritePageOutlineFromConv_EN(dspy.Signature):
    """Improve an outline for a Wikipedia page. You already have a draft outline that covers the general information. Now you want to improve it based on the information learned from an information-seeking conversation to make it more informative.
    Here is the format of your writing:
    1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
    2. Do not include additional information or explanations outside of the titles.
    3. Do not include the topic name itself in any of the titles within the outline.
    4. The outline must contain exactly {num_sections} sections, no more, no less.
    5. **The outline must be written in English.**
    """

    topic = dspy.InputField(prefix="The topic you want to write: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    old_outline = dspy.InputField(prefix="Current outline:\n", format=str)
    num_sections = dspy.InputField(prefix="Number of sections:\n", format=str)
    outline = dspy.OutputField(
        prefix=(
            'Write the Wikipedia page outline based on the conversation in English.\n'
            'Remember:\n'
            '- Use "#" Title" for sections, "##" Title" for subsections, etc.\n'
            '- Do not include the topic name in the titles.\n'
            '- Include exactly {num_sections} main sections.\n'
            'Here is the improved outline in English:\n'
        ),
        format=str,
    )