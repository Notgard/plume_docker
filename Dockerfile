FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y gcc wget
RUN wget https://dlm.mariadb.com/2678574/Connectors/c/connector-c-3.3.3/mariadb-connector-c-3.3.3-debian-bullseye-amd64.tar.gz -O - | tar -zxf - --strip-components=1 -C /usr
RUN apt-get install -y libmariadb-dev

RUN pip install --upgrade pip

RUN pip install fastapi[standard]
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && fastapi run main.py"]