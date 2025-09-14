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
mkdir -p data && curl "https://assets-datascientest.s3.eu-west-1.amazonaws.com/drift_monitoring/Delay_data.zip" -o "data/Delay_data.zip"
```

## Daily run
```bash
# Affichage des projets de démo
source .venv/bin/activate
./ui_demo_all.sh

# workspace pour regression et classification
source .venv/bin/activate
python 2_regression_monitoring.py
python 3_classification_monitoring.py
python 4_test_suites.py
python 4a_test_suites.py
./ui_workspace.sh
```

