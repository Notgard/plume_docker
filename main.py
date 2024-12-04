from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import connection, article_creation, result

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(connection.router)
app.include_router(article_creation.router)
app.include_router(result.router)
