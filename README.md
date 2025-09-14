# drift-monitoring-training
Practice drift monitoring using evidently in context of datascientest training

## Setup VM

### Initialisation des dépendances de la VM (one time)

#### Mise à jour globale des paquets et rattrapage éventuel des installations manquantes

```bash
# Mets à jour la liste des paquets
sudo apt update

# Récupère et corrige d'éventuels paquets manquants
sudo apt install --fix-missing
```

#### python3

```bash
# (Ré)installation de python3
sudo apt install -y python3
```

#### pip3 et pipx

```bash
# (Ré)installation de pip3 et pipx
sudo apt-get install -y python3-pip
sudo apt install -y pipx
pipx ensurepath
source ~/.bashrc
```

### uv

```bash
# Lance l’installation de uv
pipx install uv
```

### Check des composants nécessaire (on demand)

```bash
# Relance l’installation de pip
python3 --version
pip3 --version
pipx --version
uv --version
```

## Setup environnement virtuel et evidently (one time)

```bash
# A executer dans le repository cloné
uv sync
source .venv/bin/activate
evidently ui --demo-projects all
```