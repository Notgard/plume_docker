python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
# alembic revision --autogenerate -m "migration"
alembic upgrade head
fastapi dev main.py