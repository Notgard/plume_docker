from fastapi import APIRouter, Request, BackgroundTasks, UploadFile, File, Form
from typing import List
from fastapi.responses import HTMLResponse
from datetime import datetime
from urllib.parse import unquote
import os

from dependencies import get_models, templates, database, client
from storm import create_csv_file, create_new_article, truncate_filename
from knowledge_storm.database import Topic
from knowledge_storm.storm_wiki.engine import STORMWikiRunner

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def page_garde(request: Request):
    session = database.create_session()
    topics = session.query(Topic).all()
    files = [topic.topic for topic in topics]
    session.close()
    list_files = []
    list_files_names = []
    models = get_models()
    for i in files:
        if i[0]==".":
            continue
        else:
            list_files.append(i)
            list_files_names.append(i.replace("_", " "))
    context = {
            "request" : request,
            "list_files" : list_files,
            "list_files_names" : list_files_names,
            "models" : models
        }
    return templates.TemplateResponse(
        context = context, name = "menu.html"
    )

@router.post("/loading", response_class=HTMLResponse)
async def topic_create(
    request: Request,
    background_task: BackgroundTasks,
    topic: str = Form(...),
    nbr_paragraphe: str = Form('Pas de contrainte'),
    source: str = Form(None),
    source_llm: str = Form(None),
    language: str = Form(None),
    # langage: str = Form(None),
    files: List[UploadFile] = File(None)
    ):
    session = database.create_session()
    new_article = Topic(
        topic = topic,
        start_date = datetime.now(),
        status = "Création de l'article lancé"
        )
    session.add(new_article)
    session.commit()
    session.refresh(new_article)
    session.close()
    
    # setup du langage
    if language == "francais":
        selected_language = "fr"
    elif language == "anglais":
        selected_language = "en"
    else:
        selected_language = None  

    STORMWikiRunner.set_selected_language(selected_language)
    
    # setup des sources, pour l'instant les sources ne se cumulent pas
    path_csv_file=None
    if source == "local" :
        source_RM="local"
        if files :
            database.update_status_topic(topic, "Création du document au format .csv ...")
            path_csv_file = await create_csv_file(topic, files, client, database)
        source_RM = "local"
    elif source == "internet":
        source_RM = "internet"
    else : source_RM = "arXiv"
    model = None
    
    # setup des llm
    if source_llm == "Aristote" :
        source_LLM = "Aristote"
    else : 
        source_LLM = "local"
        model = source_llm    
    
    ###MODIF
    background_task.add_task(create_new_article, topic, source_LLM, source_RM, path_csv_file, database, nbr_paragraphe, client, model)
    ###
    topic_name_cleaned = topic.replace(' ', '_').replace('/', '_')
    topic_name_truncated = truncate_filename(topic_name_cleaned)
   
    return templates.TemplateResponse(request=request, name="loading.html", context={"topic": topic_name_truncated})

@router.get("/check_status/{topic}")
def check_status(topic: str):
    topic = unquote(topic)
    print(f"Vérification du statut pour le sujet : {topic}")
    session = database.create_session()

    try:
        # Exécuter une requête SQL synchrone
        result = session.query(Topic).filter(Topic.topic == topic).order_by(Topic.start_date.desc()).first()
        if result:
            print(f"Dernier topic trouvé: {result.topic}")
            return {"status": result.status}  # Renvoie le statut de la tâche
        else:
            print("Aucun résultat trouvé.")
            return {"status": "not found"}

    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return {"status": "error", "message": str(e)}
    
    finally:
        # Fermer la session après utilisation
        
        session.close()