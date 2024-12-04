from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
import urllib
import os
from knowledge_storm.database import Topic
import requests

from dependencies import get_file, templates, database, client

router = APIRouter()

@router.get("/result", response_class=HTMLResponse)
async def result_page(request: Request, filename: str = None):
    session = database.create_session()
    topics = session.query(Topic).all()
    files = [topic.topic for topic in topics]
    session.close()
    list_files = []
    list_files_names = []
    for i in files:
        if i[0]==".":
            continue
        else:
            list_files.append(i)
            list_files_names.append(i.replace("_", " "))
    
    if filename is None and files:
        filename = list_files[0]
    print(filename)
    print(list_files)
    
    file_content = get_file(filename, client=client)
    context = {
        "request": request,
        "file_content": file_content,
        "file_title": filename.replace("_", " "),
        "list_files": list_files,
        "list_files_names": list_files_names
    }
    return templates.TemplateResponse(context=context, name="result.html")


@router.get("/result/{filename:path}")
async def get_pdf(filename: str):
    filename = urllib.parse.unquote(filename)
    requests.get(filename)