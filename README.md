# Storm Project

# VOIR LE FICHIER "README_DEPLOY.md" POUR OBTENIR LES INFORMATIONS DE DEPLOIMENT DE PLUME

## Prérequis

 - Docker / docker compose
 - Ollama en local (si volonté de l'utiliser)

## Installation

Si vous voulez utiliser ollama, il faut lancer la commande : 

```shell
ollama pull llama3.1
```
Ensuite, pour lancer l'application, il faut lancer l'application docker et se placer à la racine du projet :

```shell
docker compose build
cp .env.dist .env
cp .env.dist.docker .env.docker
```

Il faut juste changer la valeur de `ARISTOTE_API_KEY` dans .env par la valeur de votre clé d'api Aristote dans .env et changer la valeur de `MYSQL_ROOT_PASSWORD` par celle que vous voulez dans .env.docker.
Si Ollama tourne sur un autre port que 11434, il faut également changer la variable d'environnement `OLLAMA_PORT` pour le bon port de travail.

```shell
docker compose up -d mariadb_db
docker compose up -d app
```

Une fois l'image lancée, il suffit alors de se connecter à l'adresse http://localhost:8000. 

## Paramètres du projet

Dans le cas d'absence de précision du type d'utilisation via l'interface (Ollama ou Aristote), l'utilisation par défaut du modèle se fait avec Ollama.
De même, en l'absence de précisions sur la langue, la langue de sortie sera celle détectée par l'outil dans le sujet de la recherche.

## Déploiement via Docker

### Utilisation avec Ollama

Pour utiliser Ollama, il est nécessaire d'installer à part Ollama sur votre machine. Ici le lien vers leur documentation : https://github.com/ollama/ollama. Une fois l'application installée, utilisez la commande suivante pour intaller le modèle de langage llama3.1 : 
```shell
ollama pull llama3.1
```  
Si vous souhaitez utiliser un autre modèle que llama3.1, utilisez la commande ci-dessus avec le nom de votre modèle et modifiez la variable d'environnement `OLLAMA_MODEL` dans le fichier .env.

### Utilisation avec Aristote

En cas d'utilisation avec Aristote, il faut modifier la valeur de la variable d'environnement `ARISTOTE_API_KEY` dans le fichier .env avec la valeur de votre clé d'api.

