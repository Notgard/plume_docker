#!/bin/bash

#build image
docker compose build

#copy necessary env vars
cp .env.dist .env
cp .env.dist.docker .env.docker

#start containers in proper order
docker compose up -d mariadb_db
docker compose up -d app