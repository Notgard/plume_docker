from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
import os

from dependencies import get_models, database, templates
from knowledge_storm.database import User, Topic

router = APIRouter()

@router.get("/auth", response_class=HTMLResponse)
async def page_auth(request: Request):

    session = database.create_session()
    database.check_tables()
    database.check_user()
    users = session.query(User).all()
    liste_users = []
    for user in users:
        liste_users.append(f"{user.first_name} {user.last_name}")
    
    context = {
        "request" : request,
        "list_users" : liste_users
    } 
    session.close()
    return templates.TemplateResponse(
        context=context, name="connection.html"
    )

@router.post("/")
async def get_new_user(
    request: Request,
    nom: str = Form(None),
    prenom: str = Form(None),
    user: str = Form(None)
    ):
    if prenom != None and nom != None:
        session = database.create_session()
        new_user = User(first_name = prenom, last_name = nom)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        session.close()

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
    elif user != None:
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
    else:
        return templates.TemplateResponse(
            request=request, name="connection.html"
        )