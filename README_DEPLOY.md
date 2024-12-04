# Déploiement de Plume

## Lancement de Ollama
Démarrer Ollama de la façon suivante:  
```bash
sudo docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 -e OLLAMA_DEBUG=1 -e OLLAMA_KEEP_ALIVE=30m --restart always --name ollama ollama/ollama
```

Après avoir démarré le serveur Ollama à partir de Docker, il faut maintenant faire en sorte d'avoir un modèle que l'on voudrait séléctionner pour la génération des articles. Ici, 2 façons de faire :
  
Soit en entrant directement dans le conteneur Ollama et d'effectuer ensuite le `pull` du modèle choisi :
```bash
docker exec -it ollama bash
ollama pull llama3.1
```

Ou directement en executant la commande de `pull` :
```bash
docker exec -it ollama ollama pull llama3.1
```
  
## Troubleshooting

### Détéction/Utilisation GPU

Le code source à été modifier afin que la plupart des problèmes soient réglés. Cependant des problèmes de drivers ou d'accès au GPU peuvent persister avec Ollama et Docker. Les commandes suivantes peuvent etre utilisées afin de résoudre les problèmes les plus communs : 
  
Vérifier déjà si des applications utilisent le module `nvidia-uvm` (https://github.com/ollama/ollama/blob/main/docs/gpu.md#laptop-suspend-resume) :  
```bash
lsof /dev/nvidia*
```
  
Si aucune application utilise ce module, vous pouvez alors effectuer la commande suivante :
```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm # ou sinon : sudo modprobe --remove nvidia-uvm && sudo modprobe nvidia-uvm
```

Dans le cas où des applications utilisent le module ou que la commande précédente "fail", passer par le wrapper `nvidia-modprobe` pour effectuer le rechargement du module des drivers : 
```bash
sudo nvidia-modprobe -u
```
  
Après le module rechargé, il est recommandé de redémarrer docker lui-même (et donc alors tous les conteneurs) :
```bash
docker stop ollama
sudo systemctl restart docker
docker start ollama
```

Si vous rencontrer d'autres problèmes avec cette méthode, réferé vous à la procedure suivante : https://github.com/Notgard/docker_ollama_openwebui/tree/master?tab=readme-ov-file#manually-solving-cgroupfs-problem

### Ollama/Plume

Si vous rencontrer des problèmes en utilisant les versions conteneurisés de Ollama ou Plume, vous pouvez accéder au logs de ces applications afin d'obtenir plus d'information de debogage :
```bash
docker logs -f ollama #logs d'ollama
docker logs -f fastapi-app #logs de Plume
```
---

## Lancement des conteneurs Plume

La compilation et le lancement des conteneurs à été simplifié dans le script bash `setup.sh` à la racine de ce projet :
```bash
chmod +x setup.sh && ./setup.sh
```

Une fois l'image lancée, il suffit alors de se connecter à l'adresse http://localhost:8000
Un serveur MinIO est mis à disposition afin de pouvoir stocker les logs de générations des articles Plume, accessible à http://localhost:9000 (identifiants dans l'environnement)


